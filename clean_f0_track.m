function [filtered,voiced,q] = clean_f0_track(rawF0,harmonic,rmsFrame,localPeak,p)
% Reject unsupported pitch candidates before smoothing accepted voiced runs.
rawF0=rawF0(:)'; harmonic=harmonic(:)'; rmsFrame=rmsFrame(:)'; localPeak=logical(localPeak(:)');
assert(isequal(size(rawF0),size(harmonic),size(rmsFrame),size(localPeak)), ...
    'RAVDESS:PitchShape','Pitch diagnostics must have matching frame counts.');
candidate=isfinite(rawF0) & rawF0>=p.F0RangeHz(1) & rawF0<=p.F0RangeHz(2) ...
    & isfinite(harmonic) & harmonic>=p.F0HarmonicRatioMinimum ...
    & isfinite(rmsFrame) & rmsFrame>=p.F0EnergyFraction*max(rmsFrame);
voiced=candidate & localPeak;
rejectedJump=false(size(voiced));
[first,last]=runs(voiced);
for run=1:numel(first)
    for k=first(run):last(run)
        neighbors=max(first(run),k-2):min(last(run),k+2);
        neighbors(neighbors==k)=[];
        if numel(neighbors)>=2
            reference=median(rawF0(neighbors));
            rejectedJump(k)=abs(log2(rawF0(k)/reference))>p.F0LocalDeviationOctaves;
        end
    end
end
voiced(rejectedJump)=false;
filtered=zeros(size(rawF0)); rejectedShort=false(size(voiced));
[first,last]=runs(voiced);
for run=1:numel(first)
    idx=first(run):last(run);
    if numel(idx)<p.F0MinimumRunFrames
        voiced(idx)=false; rejectedShort(idx)=true;
    else
        % Reflection supplies a complete odd window at both endpoints; it
        % avoids an even two-value median averaging a boundary outlier.
        track=rawF0(idx);
        padded=[track(2),track,track(end-1)];
        smoothed=movmedian(padded,p.F0MedianSmoothingFrames);
        filtered(idx)=smoothed(2:end-1);
    end
end
q=struct('CandidateF0Frames',sum(candidate), ...
    'RejectedNonPeakFrames',sum(candidate & ~localPeak), ...
    'RejectedJumpFrames',sum(rejectedJump),'RejectedShortRunFrames',sum(rejectedShort));
end

function [first,last]=runs(mask)
edges=diff([false mask false]); first=find(edges==1); last=find(edges==-1)-1;
end
