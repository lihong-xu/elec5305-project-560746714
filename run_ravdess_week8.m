function result = run_ravdess_week8(options)
% Prepare RAVDESS splits and course-based acoustic representations through Week 8.
if nargin < 1, options = struct; end
root = fileparts(mfilename('fullpath'));
defaults = struct('DataDir', fullfile(root,'data','raw'), ...
    'OutputDir', fullfile(root,'results'), 'DownloadData', true, ...
    'StrictDataset', true, 'NumFolds', 6, 'Seed', 5305, ...
    'Stage', 'acoustic', 'ShowFigures', false);
fields = fieldnames(defaults);
for k=1:numel(fields)
    if ~isfield(options,fields{k}), options.(fields{k})=defaults.(fields{k}); end
end
assert(ismember(string(options.Stage),["prepare","acoustic"]), ...
    'RAVDESS:Stage','Stage must be prepare or acoustic.');
assert(options.NumFolds >= 2 && fix(options.NumFolds)==options.NumFolds, ...
    'RAVDESS:Folds','NumFolds must be an integer of at least two.');
oldPath = path; oldRng = rng; oldVisible = get(groot,'defaultFigureVisible');
cleanup = onCleanup(@() restoreState(oldPath,oldRng,oldVisible)); 
courseDir = fullfile(root,'Toolbox','AudioAnalysis');
assert(isfolder(courseDir),'RAVDESS:Toolbox','The project-local AudioAnalysis folder is missing.');
addpath(courseDir,'-begin');
if options.ShowFigures, visibility='on'; else, visibility='off'; end
set(groot,'defaultFigureVisible',visibility);
out = options.OutputDir;
if ~isfolder(out), mkdir(out); end
if ~isfolder(options.DataDir) || isempty(dir(fullfile(options.DataDir,'**','*.wav')))
    if ~options.DownloadData
        error('RAVDESS:MissingData','No WAV recordings were found in DataDir.');
    end
    acquireData(root,options.DataDir);
end
manifest = buildManifest(options.DataDir,root);
if options.StrictDataset, checkComplete(manifest); end
assert(options.NumFolds <= numel(unique(manifest.ActorID)), ...
    'RAVDESS:Folds','There must be at least one actor per fold.');
rng(options.Seed,'twister');
manifest = assignSplits(manifest,options.NumFolds);
prepDir = fullfile(out,'prepared_audio');
if ~isfolder(prepDir), mkdir(prepDir); end
n = height(manifest);
manifest.PreparedPath = strings(n,1); manifest.PreparedSHA256 = strings(n,1);
for i=1:n
    [x,fs] = audioread(fullfile(root,manifest.RelativePath(i)));
    x = mean(x,2);
    if fs~=16000, x=resample(x,16000,fs); end
    assert(all(isfinite(x)) && sum(x.^2)>eps,'RAVDESS:InvalidAudio', ...
        'Selected recording %s is silent or nonfinite.',manifest.RecordingID(i));
    prepared = fullfile(prepDir,manifest.RecordingID(i)+".wav");
    needsWrite=true;
    if isfile(prepared)
        try
            [existing,existingFs]=audioread(prepared);
            needsWrite=existingFs~=16000 || ~isequal(single(existing),single(x));
        catch
            needsWrite=true;
        end
    end
    % Preserve identical float32 WAV files because their PEAK header timestamps
    % otherwise change byte digests even when the waveform is unchanged.
    if needsWrite, audiowrite(prepared,x,16000,'BitsPerSample',32); end
    manifest.PreparedPath(i)=relativePath(prepared,root);
    manifest.PreparedSHA256(i)=fileDigest(prepared,'SHA-256');
end
writetable(manifest,fullfile(out,'manifest.csv'));
emotionSplits = table(strings(0,1),zeros(0,1),zeros(0,1),strings(0,1), ...
    'VariableNames',{'RecordingID','ActorID','Fold','Partition'});
for fold=1:options.NumFolds
    partition=repmat("train",n,1); partition(manifest.EmotionFold==fold)="test";
    emotionSplits=[emotionSplits; table(manifest.RecordingID,manifest.ActorID, ...
        repmat(fold,n,1),partition,'VariableNames',emotionSplits.Properties.VariableNames)]; %#ok<AGROW>
end
writetable(emotionSplits,fullfile(out,'emotion_splits.csv'));
writetable(manifest(:,{'RecordingID','ActorID','Emotion','SpeakerSplit'}), ...
    fullfile(out,'speaker_split.csv'));
counts=groupsummary(manifest,{'ActorID','Emotion'});
writetable(counts,fullfile(out,'dataset_counts.csv'));
params=struct('SampleRateHz',16000,'WindowSamples',400,'HopSamples',160, ...
    'FFTLength',512,'Window','Hamming symmetric','MFCCCount',13, ...
    'MelFilterCount',40,'FilterSpacing','Slaney 13 linear + 27 logarithmic', ...
    'PreEmphasis',false,'F0Method','ELEC5305 feature_harmonic autocorrelation', ...
    'F0RangeHz',[62.5 500],'F0WindowSamples',800,'F0HarmonicRatioMinimum',0.6, ...
    'F0Window','Unwindowed, frame mean removed', ...
    'F0EnergyGate','pitch-frame RMS >= 0.1 * maximum pitch-frame RMS', ...
    'F0EnergyFraction',0.1,'F0MinimumRunFrames',3,'F0MedianSmoothingFrames',3, ...
    'F0LocalDeviationOctaves',0.5,'F0LocalReference','Median of up to four neighbors within 20 ms, excluding the current frame', ...
    'F0PeakCheck','Course normalized autocorrelation must have a local maximum at the selected lag', ...
    'F0MedianEndpoints','One reflected interior sample at each endpoint', ...
    'EnergyMethod','RMS of unwindowed frames; amplitude preserved', ...
    'CentroidUnit','Hz','Seed',options.Seed,'EmotionFolds',options.NumFolds, ...
    'SpeakerTestFraction',0.25,'MATLABVersion',version, ...
    'DatasetStrict',options.StrictDataset,'Stage',options.Stage);
toolNames={'getDFT','feature_mfccs_init','feature_mfccs','feature_harmonic','feature_energy'};
for k=1:numel(toolNames)
    tool=which(toolNames{k});
    assert(startsWith(tool,courseDir),'RAVDESS:ToolboxCollision','Unexpected function: %s.',tool);
    params.ToolboxFunctions.(toolNames{k})=struct('Path',relativePath(tool,root), ...
        'SHA256',fileDigest(tool,'SHA-256'));
end
writeJson(fullfile(out,'acoustic_parameters.json'),params);
dataCheck=struct('Recordings',n,'Actors',numel(unique(manifest.ActorID)), ...
    'Counts',table2struct(counts),'UnreadableRecordings',0, ...
    'EmotionActorLeakage',false,'SpeakerRecordingLeakage',false, ...
    'Source','https://zenodo.org/records/1188976');
writeJson(fullfile(out,'data_checks.json'),dataCheck);
result=struct('Manifest',manifest,'OutputDir',out);
fprintf('Prepared %d selected speech recordings from %d actors.\n',n,dataCheck.Actors);
if strcmp(options.Stage,'prepare'), return; end
featureNames = ["MFCC"+compose('%02d',1:13)+"Mean", ...
    "MFCC"+compose('%02d',1:13)+"Std", "F0MedianHz","F0RangeHz", ...
    "RMSMean","RMSStd","DurationSeconds","CentroidMeanHz","CentroidStdHz"];
features=zeros(n,numel(featureNames));
totalFrames=zeros(n,1); pitchFrames=zeros(n,1); voicedFrames=zeros(n,1);
candidateFrames=zeros(n,1); rejectedNonPeak=zeros(n,1); rejectedJump=zeros(n,1); rejectedShort=zeros(n,1);
F0Tracks=struct('RecordingID',{},'Times',{},'RawF0Hz',{},'FilteredF0Hz',{}, ...
    'HarmonicRatio',{},'FrameRMS',{},'LocalPeak',{},'Accepted',{});
peakAmplitude=zeros(n,1); clippingSamples=zeros(n,1);
for i=1:n
    [x,fs]=audioread(fullfile(root,manifest.PreparedPath(i)));
    [features(i,:),q,detail]=acousticVector(x,fs,params);
    totalFrames(i)=q.TotalFrames; pitchFrames(i)=q.TotalPitchFrames; voicedFrames(i)=q.ValidF0Frames;
    candidateFrames(i)=q.CandidateF0Frames; rejectedNonPeak(i)=q.RejectedNonPeakFrames;
    rejectedJump(i)=q.RejectedJumpFrames; rejectedShort(i)=q.RejectedShortRunFrames;
    F0Tracks(i)=struct('RecordingID',char(manifest.RecordingID(i)), ...
        'Times',detail.F0Times,'RawF0Hz',detail.RawF0,'FilteredF0Hz',detail.F0, ...
        'HarmonicRatio',detail.HarmonicRatio,'FrameRMS',detail.PitchRMS, ...
        'LocalPeak',detail.LocalPeak,'Accepted',detail.Voiced);
    peakAmplitude(i)=q.PeakAmplitude; clippingSamples(i)=q.ClippingSamples;
    if i==1, representative=detail; representative.Signal=x; end
    if mod(i,50)==0 || i==n, fprintf('Acoustic extraction: %d/%d recordings.\n',i,n); end
end
assert(all(isfinite(features),'all'),'RAVDESS:NonfiniteFeature','Acoustic features contain NaN or Inf.');
quality=table(manifest.RecordingID,totalFrames,pitchFrames,candidateFrames, ...
    rejectedNonPeak,rejectedJump,rejectedShort,voicedFrames,voicedFrames./pitchFrames, ...
    peakAmplitude,clippingSamples,'VariableNames',{'RecordingID','TotalFrames','TotalPitchFrames', ...
    'CandidateF0Frames','RejectedNonPeakFrames','RejectedJumpFrames','RejectedShortRunFrames', ...
    'ValidF0Frames','VoicedFraction','PeakAmplitude','ClippingSamples'});
RecordingID=cellstr(manifest.RecordingID); ActorID=manifest.ActorID; Emotion=cellstr(manifest.Emotion);
Features=features; FeatureNames=cellstr(featureNames); Parameters=params; 
save(fullfile(out,'acoustic_features.mat'),'Features','FeatureNames','RecordingID', ...
    'ActorID','Emotion','Parameters','-v7');
save(fullfile(out,'f0_tracks.mat'),'F0Tracks','Parameters','-v7');
featureTable=[manifest(:,{'RecordingID','ActorID','Emotion'}), ...
    array2table(features,'VariableNames',cellstr(featureNames))];
writetable(featureTable,fullfile(out,'acoustic_features.csv'));
writetable(quality,fullfile(out,'acoustic_quality.csv'));
figDir=fullfile(out,'figures'); if ~isfolder(figDir), mkdir(figDir); end
drawFigures(manifest,featureTable,representative,figDir,visibility);
check=struct('Recordings',n,'Dimensions',size(features,2),'FiniteValues',true, ...
    'MinimumVoicedFraction',min(quality.VoicedFraction),'MaximumVoicedFraction',max(quality.VoicedFraction), ...
    'RecordingsWithoutF0',sum(voicedFrames==0),'ClippingSamples',sum(clippingSamples));
check.RejectedNonPeakFrames=sum(rejectedNonPeak); check.RejectedJumpFrames=sum(rejectedJump);
check.RejectedShortRunFrames=sum(rejectedShort);
writeJson(fullfile(out,'acoustic_checks.json'),check);
result.AcousticQuality=quality; result.FeatureNames=featureNames;
fprintf('Saved %d x %d acoustic features and descriptive figures.\n',n,size(features,2));
end

function restoreState(oldPath,oldRng,oldVisible)
path(oldPath); rng(oldRng); set(groot,'defaultFigureVisible',oldVisible);
end

function acquireData(root,dataDir)
downloadDir=fullfile(root,'data','downloads'); if ~isfolder(downloadDir), mkdir(downloadDir); end
archive=fullfile(downloadDir,'Audio_Speech_Actors_01-24.zip');
url='https://zenodo.org/records/1188976/files/Audio_Speech_Actors_01-24.zip?download=1';
if ~isfile(archive)
    fprintf('Downloading the original RAVDESS speech archive.\n');
    websave(archive,url,weboptions('Timeout',180));
end
assert(strcmpi(fileDigest(archive,'MD5'),'bc696df654c87fed845eb13823edef8a'), ...
    'RAVDESS:ArchiveChecksum','The RAVDESS archive checksum does not match the official release.');
if ~isfolder(dataDir), mkdir(dataDir); end
unzip(archive,dataDir);
end

function manifest=buildManifest(dataDir,root)
files=dir(fullfile(dataDir,'**','*.wav'));
rows=cell(0,13);
for i=1:numel(files)
    id=erase(files(i).name,'.wav');
    parts=regexp(id,'^(\d{2})-(\d{2})-(\d{2})-(\d{2})-(\d{2})-(\d{2})-(\d{2})$','tokens','once');
    if isempty(parts), error('RAVDESS:Filename','Invalid WAV filename: %s.',files(i).name); end
    codes=str2double(parts);
    if codes(1)~=3 || codes(2)~=1 || ~ismember(codes(3),[1 3 4 5]), continue; end
    if ~ismember(codes(4),1:2) || ~ismember(codes(5),1:2) || ~ismember(codes(6),1:2) ...
            || ~ismember(codes(7),1:24) || (codes(3)==1 && codes(4)~=1)
        error('RAVDESS:Filename','Invalid RAVDESS label combination: %s.',id);
    end
    mapping=["neutral","","happy","sad","angry"];
    fullPath=fullfile(files(i).folder,files(i).name);
    try
        info=audioinfo(fullPath); [x,~]=audioread(fullPath);
    catch problem
        error('RAVDESS:InvalidAudio','Cannot read %s: %s.',id,problem.message);
    end
    if isempty(x) || any(~isfinite(x),'all') || sum(mean(x,2).^2)<=eps
        error('RAVDESS:InvalidAudio','Selected recording %s is silent, empty or nonfinite.',id);
    end
    rows(end+1,:)={string(id),relativePath(fullPath,root),codes(7),mapping(codes(3)), ...
        codes(3),codes(4),codes(5),codes(6),info.SampleRate,info.NumChannels, ...
        info.TotalSamples,info.Duration,fileDigest(fullPath,'SHA-256')}; %#ok<AGROW>
end
assert(~isempty(rows),'RAVDESS:EmptyDataset','No in-scope speech recordings were found.');
manifest=cell2table(rows,'VariableNames',{'RecordingID','RelativePath','ActorID','Emotion', ...
    'EmotionCode','Intensity','Statement','Repetition','OriginalSampleRate','NumChannels', ...
    'OriginalSamples','DurationSeconds','SourceSHA256'});
manifest=sortrows(manifest,'RecordingID');
assert(numel(unique(manifest.RecordingID))==height(manifest), ...
    'RAVDESS:DuplicateID','Duplicate recording IDs were found.');
end

function checkComplete(M)
if height(M)~=672 || ~isequal(unique(M.ActorID),(1:24)')
    error('RAVDESS:IncompleteDataset','The full four-emotion dataset must contain 672 recordings and 24 actors.');
end
for actor=1:24
    for code=[1 3 4 5]
        expected=8; if code==1, expected=4; end
        if sum(M.ActorID==actor & M.EmotionCode==code)~=expected
            error('RAVDESS:IncompleteDataset','Unexpected count for actor %d, emotion %d.',actor,code);
        end
    end
end
end

function M=assignSplits(M,numFolds)
actors=unique(M.ActorID); actors=actors(randperm(numel(actors)));
M.EmotionFold=zeros(height(M),1); M.SpeakerSplit=repmat("train",height(M),1);
for a=1:numel(actors), M.EmotionFold(M.ActorID==actors(a))=mod(a-1,numFolds)+1; end
for actor=unique(M.ActorID)'
    for code=unique(M.EmotionCode)'
        idx=find(M.ActorID==actor & M.EmotionCode==code);
        if numel(idx)<2, error('RAVDESS:Split','Each actor/emotion group needs at least two recordings.'); end
        idx=idx(randperm(numel(idx))); nTest=max(1,round(numel(idx)*0.25));
        M.SpeakerSplit(idx(1:nTest))="test";
    end
end
for fold=1:numFolds
    assert(isempty(intersect(unique(M.ActorID(M.EmotionFold==fold)), ...
        unique(M.ActorID(M.EmotionFold~=fold)))),'RAVDESS:Leakage','Emotion actors overlap.');
end
end

function [vector,q,detail]=acousticVector(x,fs,p)
assert(fs==p.SampleRateHz,'RAVDESS:SampleRate','Prepared audio must be 16 kHz.');
assert(numel(x)>=p.WindowSamples,'RAVDESS:ShortAudio','Audio is shorter than the analysis window.');
starts=1:p.HopSamples:(numel(x)-p.WindowSamples+1); n=numel(starts);
win=hamming(p.WindowSamples,'symmetric');
params=feature_mfccs_init(p.FFTLength+1,fs);
% The course helper initializes a full-FFT frequency grid. Map its triangular
% filters onto the actual one-sided getDFT grid before applying its MFCC DCT.
frequency=(0:p.FFTLength/2)'*fs/p.FFTLength;
weights=zeros(params.totalFilters,numel(frequency));
for k=1:params.totalFilters
    lower=params.lower(k); center=params.center(k); upper=params.upper(k);
    weights(k,:)=params.triangleHeight(k)*max(0,min((frequency-lower)/(center-lower), ...
        (upper-frequency)/(upper-center)))';
end
params.mfccFilterWeights=weights; params.fftFreqs=frequency';
mfccs=zeros(13,n); rmsFrame=zeros(1,n); centroid=zeros(1,n);
spec=zeros(numel(frequency),n);
for k=1:n
    raw=x(starts(k):starts(k)+p.WindowSamples-1); frame=raw.*win;
    [mag,~]=getDFT([frame;zeros(p.FFTLength-numel(frame),1)],fs);
    mfccs(:,k)=feature_mfccs(mag,params);
    rmsFrame(k)=sqrt(feature_energy(raw));
    if any(frame)
        centroid(k)=sum(frequency.*mag)/(sum(mag)+eps);
    end
    spec(:,k)=mag;
end
% A longer, unwindowed pitch frame retains enough periods for low voices and
% avoids Hamming attenuation bias in the course autocorrelation estimator.
pitchStarts=1:p.HopSamples:(numel(x)-p.F0WindowSamples+1);
np=numel(pitchStarts); rawF0=zeros(1,np); harmonic=rawF0; pitchRMS=rawF0; localPeak=false(1,np);
for k=1:np
    raw=x(pitchStarts(k):pitchStarts(k)+p.F0WindowSamples-1);
    pitchRMS(k)=sqrt(mean(raw.^2));
    raw=raw-mean(raw);
    if any(raw)
        [harmonic(k),rawF0(k)]=feature_harmonic(raw,fs, ...
            floor(fs/p.F0RangeHz(1)),ceil(fs/p.F0RangeHz(2)));
        if isfinite(rawF0(k)) && rawF0(k)>=p.F0RangeHz(1) && rawF0(k)<=p.F0RangeHz(2)
            lag=round(fs/rawF0(k)); correlation=zeros(1,3); fullEnergy=sum(raw.^2);
            for neighbor=1:3
                offset=lag+neighbor-2;
                left=raw(1:end-offset); right=raw(offset+1:end);
                correlation(neighbor)=sum(left.*right)/(sqrt(fullEnergy*sum(left.^2))+eps);
            end
            localPeak(k)=correlation(2)>=correlation(1) && correlation(2)>=correlation(3) ...
                && (correlation(2)>correlation(1) || correlation(2)>correlation(3));
        end
    end
end
[f0,voiced,pitchQuality]=clean_f0_track(rawF0,harmonic,pitchRMS,localPeak,p);
if ~any(voiced)
    error('RAVDESS:MissingF0','No valid voiced F0 frames were found; inspect this recording.');
end
vf=f0(voiced);
vector=[mean(mfccs,2)',std(mfccs,0,2)',median(vf),max(vf)-min(vf), ...
    mean(rmsFrame),std(rmsFrame),numel(x)/fs,mean(centroid),std(centroid)];
q=struct('TotalFrames',n,'TotalPitchFrames',np,'ValidF0Frames',sum(voiced),'PeakAmplitude',max(abs(x)), ...
    'ClippingSamples',sum(abs(x)>=1-2^-15));
for name=fieldnames(pitchQuality)', q.(name{1})=pitchQuality.(name{1}); end
detail=struct('Times',(starts-1+p.WindowSamples/2)/fs,'Frequencies',frequency, ...
    'Spectrum',spec,'MelSpectrum',weights*spec,'F0',f0,'Voiced',voiced, ...
    'RawF0',rawF0,'HarmonicRatio',harmonic,'PitchRMS',pitchRMS,'LocalPeak',localPeak, ...
    'F0Times',(pitchStarts-1+p.F0WindowSamples/2)/fs,'SampleRate',fs);
end

function drawFigures(M,T,d,figDir,visibility)
order=["neutral","happy","sad","angry"];
f=figure('Visible',visibility,'Position',[100 100 1100 750]);
layout=tiledlayout(f,2,2,'TileSpacing','compact');
fields={'F0MedianHz','F0RangeHz','RMSMean','DurationSeconds'};
labels={'F0 median (Hz)','Voiced F0 range (Hz)','Mean RMS amplitude','Duration (s)'};
for k=1:4
    nexttile(layout); group=categorical(T.Emotion,order,order,'Ordinal',true);
    boxchart(group,T.(fields{k})); ylabel(labels{k}); xlabel('Expression category'); grid on;
end
title(layout,'Preliminary acoustic distributions across acted expressions');
exportgraphics(f,fullfile(figDir,'acoustic_distributions.png'),'Resolution',150); close(f);
f=figure('Visible',visibility,'Position',[100 100 1100 700]);
layout=tiledlayout(f,2,1); nexttile(layout);
count=arrayfun(@(e)sum(M.Emotion==e),order); bar(categorical(order,order),count);
ylabel('Recordings'); title('Selected audio-only speech recordings'); grid on;
nexttile(layout); actors=unique(M.ActorID); byActor=arrayfun(@(a)sum(M.ActorID==a),actors);
bar(actors,byActor); xticks(actors); xlabel('Actor ID'); ylabel('Recordings'); grid on;
exportgraphics(f,fullfile(figDir,'dataset_counts.png'),'Resolution',150); close(f);
f=figure('Visible',visibility,'Position',[100 100 1100 950]); layout=tiledlayout(f,4,1);
nexttile(layout); plot((0:numel(d.Signal)-1)/d.SampleRate,d.Signal); ylabel('Amplitude'); grid on;
title('Representative recording: '+M.RecordingID(1),'Interpreter','none');
nexttile(layout); imagesc(d.Times,d.Frequencies,20*log10(d.Spectrum+eps)); axis xy;
ylabel('Frequency (Hz)'); title('Hamming-window STFT magnitude (dB)'); colorbar;
nexttile(layout); imagesc(d.Times,1:size(d.MelSpectrum,1),20*log10(d.MelSpectrum+eps)); axis xy;
ylabel('Mel filter index'); title('Course-based Mel filterbank magnitude (dB)'); colorbar;
nexttile(layout); plot(d.F0Times(d.Voiced),d.F0(d.Voiced),'.'); ylabel('F0 (Hz)');
xlabel('Time (s)'); title('Accepted voiced-frame fundamental frequency'); grid on;
exportgraphics(f,fullfile(figDir,'representative_signal.png'),'Resolution',150); close(f);
end

function relative=relativePath(fullPath,root)
fullPath=char(java.io.File(char(fullPath)).getCanonicalPath());
root=char(java.io.File(root).getCanonicalPath());
assert(startsWith(lower(fullPath),lower([root filesep])),'RAVDESS:Path', ...
    'Data and output directories must be inside the project root.');
relative=string(strrep(fullPath(numel(root)+2:end),filesep,'/'));
end

function hex=fileDigest(file,algorithm)
md=java.security.MessageDigest.getInstance(algorithm);
fid=fopen(file,'rb'); assert(fid>=0,'RAVDESS:Read','Cannot open file: %s.',file);
cleanup=onCleanup(@()fclose(fid)); 
while true
    bytes=fread(fid,8*1024*1024,'*uint8'); if isempty(bytes), break; end
    md.update(typecast(bytes,'int8'));
end
hex=string(lower(reshape(dec2hex(typecast(md.digest(),'uint8'),2)',1,[])));
end

function writeJson(file,value)
fid=fopen(file,'w','n','UTF-8'); assert(fid>=0,'RAVDESS:Write','Cannot write %s.',file);
cleanup=onCleanup(@()fclose(fid)); 
fprintf(fid,'%s\n',jsonencode(value,'PrettyPrint',true));
end
