"""Extract frozen WavLM transformer layers from the MATLAB Week 8 manifest."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODEL = "microsoft/wavlm-base-plus"
LAYERS = [1, 4, 8, 12]
EMOTIONS = {1: "neutral", 3: "happy", 4: "sad", 5: "angry"}


def parse_recording_id(recording_id: str) -> dict:
    if not re.fullmatch(r"\d{2}(?:-\d{2}){6}", recording_id):
        raise ValueError(f"Invalid recording ID: {recording_id}")
    modality, channel, emotion, intensity, statement, repetition, actor = map(int, recording_id.split("-"))
    if (modality != 3 or channel != 1 or emotion not in EMOTIONS or intensity not in [1, 2]
            or statement not in [1, 2] or repetition not in [1, 2] or not 1 <= actor <= 24
            or (emotion == 1 and intensity != 1)):
        raise ValueError(f"Out-of-scope or invalid recording ID: {recording_id}")
    return {"actor": actor, "emotion": EMOTIONS[emotion], "emotion_code": emotion}


def pool_hidden_states(hidden_states, layers: list[int]) -> np.ndarray:
    vectors = []
    for layer in layers:
        if not 1 <= layer <= 12 or layer >= len(hidden_states):
            raise ValueError("Layer indices must refer to transformer blocks 1 through 12.")
        state = hidden_states[layer]
        if hasattr(state, "detach"):
            state = state.detach().float().cpu().numpy()
        state = np.asarray(state)
        if state.ndim != 3 or state.shape[0] != 1 or state.shape[1] == 0 or state.shape[2] != 768:
            raise ValueError(f"Unexpected hidden-state shape: {state.shape}")
        vectors.append(state[0].mean(axis=0, dtype=np.float32))
    result = np.stack(vectors).astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError("A pooled embedding contains NaN or Inf.")
    return result


def save_npz_atomic(path: Path, **values) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **values)
    os.replace(temporary, path)


def validate_cache(cache, fingerprint: str, ids, layers) -> None:
    if str(cache["fingerprint"].item()) != fingerprint:
        raise ValueError("Stale cache: model, waveform, manifest or environment fingerprint changed.")
    if not np.array_equal(cache["recording_ids"], np.asarray(ids)):
        raise ValueError("Cached recording IDs do not match the manifest order.")
    if not np.array_equal(cache["layers"], np.asarray(layers)):
        raise ValueError("Cached transformer layers do not match the requested layers.")
    if cache["embeddings"].shape != (len(ids), len(layers), 768):
        raise ValueError("Invalid embedding cache dimensions.")
    done = np.asarray(cache["completed"])
    if done.shape != (len(ids),) or done.dtype != np.bool_:
        raise ValueError("Invalid completed-recording mask.")
    if not np.isfinite(cache["embeddings"][done]).all():
        raise ValueError("A completed cached embedding contains NaN or Inf.")


def retire_final_outputs(output: Path, reason: str) -> None:
    names = ['embeddings.npz', 'embeddings.mat', 'embedding_index.csv',
             'extraction_metadata.json', 'model_source.json']
    if not any((output / name).exists() for name in names[:4]):
        return
    archive = output / 'archive' / str(time.time_ns())
    archive.mkdir(parents=True)
    for name in names:
        source = output / name
        if source.exists():
            source.rename(archive / name)
    write_json(archive / 'archive_reason.json', {'reason': reason})
    print(f'Previous final outputs preserved under the output directory in {archive.relative_to(output)}: {reason}.', flush=True)


def prepare_cache_state(output: Path, fingerprint: str, ids, layers):
    embeddings = np.zeros((len(ids), len(layers), 768), np.float32)
    completed = np.zeros(len(ids), bool)
    checkpoint = output / 'embeddings_checkpoint.npz'
    if checkpoint.exists():
        try:
            with np.load(checkpoint, allow_pickle=False) as cache:
                validate_cache(cache, fingerprint, ids, layers)
                embeddings[:] = cache['embeddings']
                completed[:] = cache['completed']
        except Exception:
            retire_final_outputs(output, 'checkpoint rejected as stale or invalid')
            raise
        print(f'Resuming {completed.sum()}/{len(ids)} completed recordings.', flush=True)
    if not completed.all():
        retire_final_outputs(output, 'new or partial extraction has not completed')
    return embeddings, completed


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def project_path(relative: str) -> Path:
    path = (ROOT / relative).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError(f"Path escapes the project root: {relative}")
    return path


def load_manifest(manifest: Path) -> list[dict]:
    with manifest.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or len({row["RecordingID"] for row in rows}) != len(rows):
        raise ValueError("The manifest is empty or contains duplicate recording IDs.")
    expected = {f'03-01-{emotion:02d}-{intensity:02d}-{statement:02d}-{repetition:02d}-{actor:02d}'
                for actor in range(1, 25) for emotion in EMOTIONS
                for intensity in ([1] if emotion == 1 else [1, 2])
                for statement in [1, 2] for repetition in [1, 2]}
    if [row['RecordingID'] for row in rows] != sorted(expected):
        raise ValueError('Week 8 requires the canonical sorted 672-recording manifest; use --limit for a pilot.')
    for row in rows:
        labels = parse_recording_id(row["RecordingID"])
        if labels["actor"] != int(row["ActorID"]) or labels["emotion"] != row["Emotion"]:
            raise ValueError(f"Manifest labels disagree with the filename: {row['RecordingID']}")
        waveform = project_path(row["PreparedPath"])
        if sha256(waveform) != row["PreparedSHA256"]:
            raise ValueError(f"Prepared waveform digest changed: {row['RecordingID']}")
    return rows


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "results" / "manifest.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "wavlm")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--limit", type=int, help="Pilot only; partial runs never produce a final cache.")
    parser.add_argument("--revision", help="Immutable Hub commit SHA; default resolves once and saves it.")
    args = parser.parse_args()
    if args.threads < 1 or (args.limit is not None and args.limit < 1):
        parser.error("Threads and pilot limit must be positive.")
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))
    import soundfile as sf
    import torch
    from huggingface_hub import HfApi
    from scipy.io import savemat
    from transformers import WavLMModel, Wav2Vec2FeatureExtractor

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(5305)
    torch.use_deterministic_algorithms(True)
    try:
        rows = load_manifest(args.manifest)
    except Exception:
        retire_final_outputs(args.output, 'manifest or prepared waveform validation failed')
        raise
    ids = np.asarray([row["RecordingID"] for row in rows])
    actors = np.asarray([int(row["ActorID"]) for row in rows], np.int32)
    emotions = np.asarray([row["Emotion"] for row in rows])
    args.output.mkdir(parents=True, exist_ok=True)
    source_file = args.output / "model_source.json"
    if args.revision:
        revision = args.revision
    elif source_file.exists():
        revision = json.loads(source_file.read_text())["revision"]
    else:
        revision = HfApi().model_info(MODEL).sha
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("The model revision must be an immutable 40-character commit SHA.")
    packages = {name: version(name) for name in ["torch", "transformers", "huggingface-hub", "numpy", "scipy", "soundfile"]}
    parameters = {"model": MODEL, "revision": revision, "layers": LAYERS,
                  "sample_rate_hz": 16000, "pooling": "mean over all unpadded model time frames",
                  "dimension": 768, "dtype": "float32", "device": "cpu", "threads": args.threads,
                  "seed": 5305, "software": packages, "manifest_sha256": sha256(args.manifest)}
    fingerprint = hashlib.sha256(json.dumps(parameters, sort_keys=True).encode()).hexdigest()
    checkpoint = args.output / "embeddings_checkpoint.npz"
    embeddings, completed = prepare_cache_state(args.output, fingerprint, ids, LAYERS)
    write_json(source_file, {"model": MODEL, "revision": revision,
                            "source": f"https://huggingface.co/{MODEL}/tree/{revision}"})
    print(f"Loading actual frozen {MODEL} at {revision}.", flush=True)
    started = time.perf_counter()
    processor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL, revision=revision)
    model = WavLMModel.from_pretrained(MODEL, revision=revision)
    model.eval()
    model.requires_grad_(False)
    if model.config.hidden_size != 768 or model.config.num_hidden_layers != 12:
        raise ValueError("The loaded model is not the required WavLM Base+ architecture.")
    parameters["processor_config"] = processor.to_dict()
    parameters["frozen_parameters"] = all(not p.requires_grad for p in model.parameters())
    parameters["training_mode"] = model.training
    parameters["parameter_count"] = sum(p.numel() for p in model.parameters())
    limit = min(args.limit or len(rows), len(rows))
    processed = 0

    def checkpoint_save():
        save_npz_atomic(checkpoint, embeddings=embeddings, completed=completed,
                        recording_ids=ids, actors=actors, emotions=emotions,
                        layers=np.asarray(LAYERS), fingerprint=np.asarray(fingerprint))

    try:
        with torch.inference_mode():
            for i, row in enumerate(rows[:limit]):
                if completed[i]:
                    continue
                audio, sample_rate = sf.read(project_path(row["PreparedPath"]), dtype="float32")
                if sample_rate != 16000 or audio.ndim != 1 or not np.isfinite(audio).all() or not np.any(audio):
                    raise ValueError(f"Invalid prepared waveform: {ids[i]}")
                inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
                output = model(**inputs, output_hidden_states=True)
                embeddings[i] = pool_hidden_states(output.hidden_states, LAYERS)
                completed[i] = True
                processed += 1
                del output, inputs
                if processed % 10 == 0 or i == limit - 1:
                    checkpoint_save()
                    print(f"WavLM {completed.sum()}/{len(rows)}; elapsed {time.perf_counter()-started:.1f} s.", flush=True)
    finally:
        checkpoint_save()
    parameters["elapsed_seconds_this_run"] = time.perf_counter() - started
    parameters["completed_recordings"] = int(completed.sum())
    parameters["total_recordings"] = len(rows)
    parameters["fingerprint"] = fingerprint
    if not completed.all():
        write_json(args.output / "pilot_metadata.json", parameters)
        print("Pilot checkpoint saved; full Week 8 cache is not complete.", flush=True)
        return
    validate_cache({"embeddings": embeddings, "completed": completed, "recording_ids": ids,
                    "layers": np.array(LAYERS), "fingerprint": np.array(fingerprint)}, fingerprint, ids, LAYERS)
    save_npz_atomic(args.output / "embeddings.npz", embeddings=embeddings,
                    recording_ids=ids, actors=actors, emotions=emotions, layers=np.array(LAYERS),
                    fingerprint=np.array(fingerprint), completed=completed)
    temporary = args.output / "embeddings.mat.tmp"
    savemat(str(temporary), {"Embeddings": embeddings, "RecordingID": ids.astype(object),
                            "ActorID": actors[:, None], "Emotion": emotions.astype(object),
                            "Layers": np.array(LAYERS), "Fingerprint": fingerprint},
            appendmat=False, do_compression=True)
    os.replace(temporary, args.output / "embeddings.mat")
    write_json(args.output / "extraction_metadata.json", parameters)
    cache_manifest = args.output / "embedding_index.csv"
    with cache_manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["RowIndex", "RecordingID", "ActorID", "Emotion", "Layer", "Dimensions"])
        for i, row in enumerate(rows):
            for layer in LAYERS:
                writer.writerow([i, row["RecordingID"], row["ActorID"], row["Emotion"], layer, 768])
    print(f"COMPLETE: actual frozen WavLM cache shape {embeddings.shape}.", flush=True)


if __name__ == "__main__":
    main()
