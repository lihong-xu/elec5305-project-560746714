"""Evaluate frozen representations with separate nested emotion/speaker probes."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import tempfile
import time
from importlib.metadata import version
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
CLASSES = ['neutral', 'happy', 'sad', 'angry']
LAYERS = [1, 4, 8, 12]
C_GRID = [0.001, 0.01, 0.1, 1.0, 10.0]
SEED = 5305


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    rows = list(rows)
    if not rows:
        raise ValueError('Cannot write an empty result table.')
    with Path(path).open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def digest(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_inputs(results):
    manifest = read_csv(results / 'manifest.csv')
    ids = np.array([row['RecordingID'] for row in manifest])
    actors = np.array([int(row['ActorID']) for row in manifest])
    y = np.array([row['Emotion'] for row in manifest])
    require(len(ids) == 672 and len(set(ids)) == 672, 'Expected 672 unique recordings.')
    require(set(actors) == set(range(1, 25)), 'Expected all 24 actors.')
    a = loadmat(results / 'acoustic_features.mat', simplify_cells=True)
    require(np.array_equal(a['RecordingID'], ids), 'Acoustic IDs differ from manifest.')
    require(np.array_equal(a['ActorID'], actors) and np.array_equal(a['Emotion'], y), 'Acoustic labels differ.')
    acoustic_rows = read_csv(results / 'acoustic_features.csv')
    require([r['RecordingID'] for r in acoustic_rows] == list(ids), 'Acoustic CSV order differs.')
    csv_features = np.array([[float(r[n]) for n in a['FeatureNames']] for r in acoustic_rows])
    require(np.allclose(csv_features, a['Features'], rtol=1e-12, atol=1e-12), 'Acoustic CSV/MAT differ.')
    representations = {'Acoustic': np.asarray(a['Features'], dtype=float)}
    require(representations['Acoustic'].shape == (672, 33), 'Expected 33 acoustic features.')
    with np.load(results / 'wavlm/embeddings.npz', allow_pickle=False) as cache:
        require(np.array_equal(cache['recording_ids'], ids), 'WavLM IDs differ.')
        require(np.array_equal(cache['actors'], actors) and np.array_equal(cache['emotions'], y), 'WavLM labels differ.')
        require(np.array_equal(cache['layers'], LAYERS) and cache['completed'].all(), 'Incomplete WavLM cache.')
        require(cache['embeddings'].shape == (672, 4, 768), 'Incorrect WavLM dimensions.')
        for slot, layer in enumerate(LAYERS):
            representations[f'WavLM_L{layer:02d}'] = cache['embeddings'][:, slot].astype(float)
    metadata = json.loads((results / 'wavlm/extraction_metadata.json').read_text())
    require(metadata['manifest_sha256'] == digest(results / 'manifest.csv'), 'Stale WavLM manifest.')
    require(metadata['frozen_parameters'] and metadata['training_mode'] is False, 'Model was not frozen.')
    require(all(np.isfinite(x).all() for x in representations.values()), 'Nonfinite input features.')
    folds = np.array([int(row['EmotionFold']) for row in manifest])
    require(set(folds) == set(range(1, 7)), 'Expected six outer folds.')
    split_rows = read_csv(results / 'emotion_splits.csv')
    require(len(split_rows) == 672 * 6, 'Incomplete emotion assignment table.')
    for fold in range(1, 7):
        rows = [r for r in split_rows if int(r['Fold']) == fold]
        require(len(rows) == 672 and {r['RecordingID'] for r in rows} == set(ids), 'Invalid emotion IDs.')
        lookup = {r['RecordingID']: r for r in rows}
        for i, identifier in enumerate(ids):
            require(int(lookup[identifier]['ActorID']) == actors[i], 'Split actor mismatch.')
            require(lookup[identifier]['Partition'] == ('test' if folds[i] == fold else 'train'), 'Split mismatch.')
        train, test = folds != fold, folds == fold
        require(test.sum() == 112 and not set(actors[train]) & set(actors[test]), 'Outer actor leakage.')
    speaker_rows = read_csv(results / 'speaker_split.csv')
    require([r['RecordingID'] for r in speaker_rows] == list(ids), 'Speaker IDs differ.')
    speaker_train = np.array([r['SpeakerSplit'] == 'train' for r in speaker_rows])
    for i, row in enumerate(speaker_rows):
        require(row['SpeakerSplit'] in ['train', 'test'] and row['SpeakerSplit'] == manifest[i]['SpeakerSplit'], 'Speaker split differs.')
        require(int(row['ActorID']) == actors[i] and row['Emotion'] == y[i], 'Speaker labels differ.')
    for actor in range(1, 25):
        require(np.sum(speaker_train & (actors == actor)) == 21 and np.sum(~speaker_train & (actors == actor)) == 7,
                'Incorrect per-actor speaker counts.')
    return manifest, ids, actors, y, representations, folds, speaker_train, a


def inner_folds(y, actors, task, seed):
    if task == 'emotion':
        splits = list(StratifiedGroupKFold(3, shuffle=True, random_state=seed).split(np.zeros(len(y)), y, actors))
    else:
        splits = list(StratifiedKFold(3, shuffle=True, random_state=seed).split(np.zeros(len(y)), y))
    for train, valid in splits:
        require(not set(train) & set(valid), 'Inner utterance overlap.')
        require(set(y[train]) == set(y) and set(y[valid]) == set(y), 'Missing inner class.')
        if task == 'emotion':
            require(not set(actors[train]) & set(actors[valid]), 'Inner emotion actor leakage.')
    return splits


def tune_model(x, y, splits, task):
    # The scaler is fitted anew inside each inner training fold and on refit.
    estimator = Pipeline([('scale', StandardScaler()),
                          ('svm', SVC(kernel='linear', class_weight='balanced', random_state=SEED))])
    search = GridSearchCV(estimator, {'svm__C': C_GRID}, cv=splits,
                          scoring='f1_macro' if task == 'emotion' else 'accuracy',
                          n_jobs=1, refit=True, error_score='raise', return_train_score=False)
    search.fit(x, y)
    return search


def metrics(y, pred):
    return {'MacroF1': float(f1_score(y, pred, average='macro', zero_division=0)),
            'Accuracy': float(accuracy_score(y, pred))}


def save_class_results(out, task, name, y, pred, labels):
    p, r, f, n = precision_recall_fscore_support(y, pred, labels=labels, zero_division=0)
    records = [dict(Task=task, Representation=name, Class=str(label), Precision=float(p[i]),
                    Recall=float(r[i]), F1=float(f[i]), Support=int(n[i])) for i, label in enumerate(labels)]
    cm = confusion_matrix(y, pred, labels=labels)
    write_csv(out / f'{task}_confusion_{name}.csv',
              [dict(TrueClass=str(label), **{f'Predicted_{k}': int(v) for k, v in zip(labels, cm[i])})
               for i, label in enumerate(labels)])
    return records, cm


def draw_confusions(folder, task, cms, best, labels):
    fig, axes = plt.subplots(2, 2, figsize=(11, 9), constrained_layout=True)
    for column, name in enumerate(['Acoustic', best]):
        cm = cms[name]
        for row in range(2):
            ax = axes[row, column]
            data = cm if row == 0 else cm / cm.sum(axis=1, keepdims=True)
            common_max = max(cms['Acoustic'].max(), cms[best].max())
            im = ax.imshow(data, cmap='Blues', vmin=0, vmax=common_max if row == 0 else 1)
            ax.set(title=f'{name}: ' + ('counts' if row == 0 else 'row-normalized'),
                   xlabel='Predicted class', ylabel='True class')
            ax.set_xticks(range(len(labels)), labels, rotation=45, ha='right')
            ax.set_yticks(range(len(labels)), labels)
            ax.tick_params(labelsize=8 if len(labels) == 24 else 10)
            if len(labels) <= 4:
                for i in range(len(labels)):
                    for j in range(len(labels)):
                        ax.text(j, i, str(cm[i, j]) if row == 0 else f'{data[i,j]:.2f}',
                                ha='center', va='center', color='white' if data[i,j] > data.max()/2 else 'black')
            fig.colorbar(im, ax=ax, shrink=.75)
    fig.suptitle(f'{task.capitalize()} probe: baseline and descriptive best emotion layer')
    fig.savefig(folder / f'{task}_confusion_comparison.png', dpi=180)
    plt.close(fig)


def run(results, out):
    manifest, ids, actors, y, representations, folds, speaker_train, acoustic = load_inputs(results)
    out.mkdir(parents=True, exist_ok=True)
    figures = out / 'figures'; figures.mkdir(exist_ok=True)
    predictions, fold_metrics, tuning, inner_rows, class_rows = [], [], [], [], []
    summaries, emotion_cms, speaker_cms = [], {}, {}
    all_emotion_predictions, layer_inner_scores = {}, {}
    for fold in range(1, 7):
        indices = np.flatnonzero(folds != fold)
        splits = inner_folds(y[indices], actors[indices], 'emotion', SEED + fold)
        for inner, (train, valid) in enumerate(splits, 1):
            for part, local in [('train', train), ('validation', valid)]:
                inner_rows.extend(dict(Task='emotion', OuterFold=fold, InnerFold=inner,
                                       Partition=part, RecordingID=ids[i], ActorID=int(actors[i])) for i in indices[local])
    speaker_indices = np.flatnonzero(speaker_train)
    speaker_cv = inner_folds(actors[speaker_train], actors[speaker_train], 'speaker', SEED)
    for inner, (train, valid) in enumerate(speaker_cv, 1):
        for part, local in [('train', train), ('validation', valid)]:
            inner_rows.extend(dict(Task='speaker', OuterFold=0, InnerFold=inner, Partition=part,
                                   RecordingID=ids[i], ActorID=int(actors[i])) for i in speaker_indices[local])
    for name, x in representations.items():
        pred = np.full(len(ids), '', dtype='<U16')
        layer_inner_scores[name] = []
        scores = []
        for fold in range(1, 7):
            train, test = folds != fold, folds == fold
            cv = inner_folds(y[train], actors[train], 'emotion', SEED + fold)
            model = tune_model(x[train], y[train], cv, 'emotion')
            pred[test] = model.predict(x[test])
            score = metrics(y[test], pred[test]); scores.append(score)
            fold_metrics.append(dict(Representation=name, Fold=fold, TrainCount=int(train.sum()),
                                     TestCount=int(test.sum()), BestC=float(model.best_params_['svm__C']),
                                     InnerMacroF1=float(model.best_score_), **score))
            layer_inner_scores[name].append(float(model.best_score_))
            for c, mean, sd in zip(model.cv_results_['param_svm__C'], model.cv_results_['mean_test_score'], model.cv_results_['std_test_score']):
                tuning.append(dict(Task='emotion', Representation=name, OuterFold=fold, C=float(c),
                                   MeanValidationScore=float(mean), StdValidationScore=float(sd)))
            print(f'{name}: emotion fold {fold}/6, macro-F1={score["MacroF1"]:.4f}', flush=True)
        all_emotion_predictions[name] = pred.copy()
        predictions.extend(dict(Representation=name, RecordingID=ids[i], ActorID=int(actors[i]), Fold=int(folds[i]),
                                TrueLabel=y[i], PredictedLabel=pred[i]) for i in range(len(ids)))
        classes, emotion_cms[name] = save_class_results(out, 'emotion', name, y, pred, CLASSES)
        class_rows.extend(classes)
        speaker_model = tune_model(x[speaker_train], actors[speaker_train], speaker_cv, 'speaker')
        speaker_pred = speaker_model.predict(x[~speaker_train])
        speaker_score = metrics(actors[~speaker_train], speaker_pred)
        write_csv(out / f'speaker_predictions_{name}.csv',
                  [dict(Representation=name, RecordingID=ids[i], TrueActor=int(actors[i]), PredictedActor=int(p))
                   for i, p in zip(np.flatnonzero(~speaker_train), speaker_pred)])
        classes, speaker_cms[name] = save_class_results(out, 'speaker', name, actors[~speaker_train], speaker_pred, list(range(1, 25)))
        class_rows.extend(classes)
        for c, mean, sd in zip(speaker_model.cv_results_['param_svm__C'], speaker_model.cv_results_['mean_test_score'], speaker_model.cv_results_['std_test_score']):
            tuning.append(dict(Task='speaker', Representation=name, OuterFold=0, C=float(c),
                               MeanValidationScore=float(mean), StdValidationScore=float(sd)))
        summaries.append(dict(Representation=name, Dimensions=x.shape[1],
                              EmotionPooledMacroF1=metrics(y, pred)['MacroF1'], EmotionPooledAccuracy=metrics(y, pred)['Accuracy'],
                              EmotionMeanFoldMacroF1=float(np.mean([s['MacroF1'] for s in scores])),
                              EmotionStdFoldMacroF1=float(np.std([s['MacroF1'] for s in scores], ddof=1)),
                              SpeakerAccuracy=speaker_score['Accuracy'], SpeakerMacroF1=speaker_score['MacroF1'],
                              SpeakerBestC=float(speaker_model.best_params_['svm__C']), SpeakerInnerAccuracy=float(speaker_model.best_score_)))
        print(f'{name}: separate speaker accuracy={speaker_score["Accuracy"]:.4f}', flush=True)
    layers = [f'WavLM_L{k:02d}' for k in LAYERS]
    best = max(layers, key=lambda name: next(s['EmotionMeanFoldMacroF1'] for s in summaries if s['Representation'] == name))
    selected = np.full(len(ids), '', dtype='<U16'); selections = []
    for fold in range(1, 7):
        winner = max(layers, key=lambda name: layer_inner_scores[name][fold - 1])
        selected[folds == fold] = all_emotion_predictions[winner][folds == fold]
        selections.append(dict(Fold=fold, SelectedRepresentation=winner,
                               InnerMacroF1=layer_inner_scores[winner][fold - 1],
                               **metrics(y[folds == fold], selected[folds == fold])))
    write_csv(out / 'training_selected_layer_folds.csv', selections)
    write_csv(out / 'training_selected_layer_predictions.csv',
              [dict(RecordingID=ids[i], Fold=int(folds[i]), TrueLabel=y[i], PredictedLabel=selected[i]) for i in range(len(ids))])
    write_csv(out / 'representation_summary.csv', summaries)
    write_csv(out / 'emotion_fold_metrics.csv', fold_metrics)
    write_csv(out / 'emotion_predictions.csv', predictions)
    write_csv(out / 'class_metrics.csv', class_rows)
    write_csv(out / 'hyperparameter_search.csv', tuning)
    write_csv(out / 'inner_splits.csv', inner_rows)
    draw_confusions(figures, 'emotion', emotion_cms, best, CLASSES)
    draw_confusions(figures, 'speaker', speaker_cms, best, [str(k) for k in range(1, 25)])
    plot_comparisons(figures, summaries, acoustic, manifest)
    metadata = dict(Status='COMPLETE', Classifier='StandardScaler + SVC(kernel=linear, class_weight=balanced)',
                    CGrid=C_GRID, Seed=SEED, EmotionOuterFolds=6, EmotionInnerFolds=3,
                    EmotionSelection='Training-only actor-grouped macro-F1',
                    SpeakerTrain=504, SpeakerTest=168, SpeakerInnerFolds=3,
                    SpeakerSelection='Training-only stratified actor accuracy',
                    DescriptiveBestLayer=best, BestLayerCriterion='Mean outer-fold emotion macro-F1; smaller layer wins ties',
                    WinnerSelectionCaveat='Winner selected using these outer results; not independent confirmation.',
                    TrainingSelectedLayerMetrics=metrics(y, selected),
                    TrainingSelectedLayerInterpretation='Layer and C selected inside each outer training set only.',
                    Software={p: version(p) for p in ['numpy','scipy','scikit-learn','matplotlib','threadpoolctl']},
                    Inputs={str(p.relative_to(ROOT)).replace('\\','/'): digest(p) for p in
                            [results/'manifest.csv', results/'emotion_splits.csv', results/'speaker_split.csv',
                             results/'acoustic_features.mat', results/'acoustic_features.csv',
                             results/'wavlm/embeddings.npz', results/'wavlm/extraction_metadata.json']},
                    SourceSHA256=digest(Path(__file__)),
                    Uncertainty='Fold standard deviation is descriptive, not a confidence interval. Speaker accuracy is one fixed-split estimate.')
    (out / 'probe_metadata.json').write_text(json.dumps(metadata, indent=2)+'\n', encoding='utf-8')
    output_hashes = {p.relative_to(out).as_posix(): digest(p) for p in sorted(out.rglob('*')) if p.is_file() and p.name != 'output_manifest.json'}
    (out/'output_manifest.json').write_text(json.dumps(output_hashes, indent=2)+'\n')
    print(f'COMPLETE: 5 representations, 30 emotion folds, 5 independent speaker probes. Best descriptive layer: {best}.', flush=True)


def plot_comparisons(folder, summaries, acoustic, manifest):
    baseline, layers = summaries[0], summaries[1:]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    axes[0].errorbar(LAYERS, [s['EmotionMeanFoldMacroF1'] for s in layers],
                     yerr=[s['EmotionStdFoldMacroF1'] for s in layers], marker='o', capsize=4, label='WavLM: mean +/- fold SD')
    axes[0].axhline(baseline['EmotionMeanFoldMacroF1'], linestyle='--', color='gray', label='Acoustic baseline')
    axes[0].set(xlabel='Transformer layer', ylabel='Emotion macro-F1', title='Actor-independent emotion probe', ylim=(0,1))
    axes[1].plot(LAYERS, [s['SpeakerAccuracy'] for s in layers], 'o-', label='WavLM')
    axes[1].axhline(baseline['SpeakerAccuracy'], linestyle='--', color='gray', label='Acoustic baseline')
    axes[1].axhline(1/24, linestyle=':', color='black', label='Uniform guess: 1/24')
    axes[1].set(xlabel='Transformer layer', ylabel='Speaker accuracy', title='Separate identity probe', ylim=(0,1.03))
    for ax in axes:
        ax.set_xticks(LAYERS); ax.grid(alpha=.25); ax.legend(fontsize=8)
    fig.savefig(folder/'layer_performance.png', dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7,5), constrained_layout=True)
    label_offsets = {'Acoustic':(10,-12),'WavLM_L01':(10,-16),'WavLM_L04':(10,18),
                     'WavLM_L08':(-85,18),'WavLM_L12':(-95,-15)}
    for s in summaries:
        ax.scatter(s['SpeakerAccuracy'], s['EmotionMeanFoldMacroF1'], s=65)
        ax.annotate(s['Representation'], (s['SpeakerAccuracy'], s['EmotionMeanFoldMacroF1']),
                    xytext=label_offsets[s['Representation']], textcoords='offset points', fontsize=9,
                    arrowprops={'arrowstyle':'-', 'color':'gray', 'lw':.7})
    ax.set(xlabel='Speaker accuracy (separate test split)', ylabel='Emotion macro-F1 (mean outer fold)',
           title='Observed expression and identity accessibility', xlim=(0,1.15), ylim=(0,1))
    ax.grid(alpha=.25); fig.savefig(folder/'emotion_speaker_relationship.png', dpi=180); plt.close(fig)
    names=list(acoustic['FeatureNames']); features=acoustic['Features']; emotion=np.array([r['Emotion'] for r in manifest])
    fields=[('F0MedianHz','F0 median (Hz)'),('F0RangeHz','Voiced F0 range (Hz)'),('RMSMean','Mean RMS amplitude'),
            ('DurationSeconds','Duration (s)'),('CentroidMeanHz','Mean spectral centroid (Hz)')]
    rows=[]; fig, axes=plt.subplots(2,3,figsize=(13,7),constrained_layout=True)
    for ax,(name,label) in zip(axes.flat,fields):
        groups=[features[emotion==c,names.index(name)] for c in CLASSES]
        ax.boxplot(groups,tick_labels=CLASSES); ax.set(ylabel=label); ax.grid(axis='y',alpha=.25)
        for c,v in zip(CLASSES,groups):
            rows.append(dict(Feature=name,Expression=c,N=len(v),Median=float(np.median(v)),
                             Q1=float(np.quantile(v,.25)),Q3=float(np.quantile(v,.75)),Mean=float(v.mean()),Std=float(v.std(ddof=1))))
    axes.flat[-1].axis('off'); fig.suptitle('Acoustic properties of four acted expression categories')
    fig.savefig(folder/'acoustic_distributions_complete.png',dpi=180); plt.close(fig)
    write_csv(folder.parent/'acoustic_descriptive_statistics.csv',rows)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results',type=Path,default=ROOT/'results')
    parser.add_argument('--output',type=Path,default=ROOT/'results/probes')
    args=parser.parse_args()
    # Publish only a completed run. Failed evaluations leave existing outputs
    # intact, and successful replacement preserves the preceding run locally.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.probe_run_', dir=args.output.parent) as staging:
        staged=Path(staging)/'probes'
        with threadpool_limits(limits=1):
            run(args.results,staged)
        if args.output.exists():
            archive=args.output.parent/'probe_archives'/str(time.time_ns())
            archive.parent.mkdir(exist_ok=True)
            args.output.rename(archive)
        staged.replace(args.output)


if __name__=='__main__':
    main()
