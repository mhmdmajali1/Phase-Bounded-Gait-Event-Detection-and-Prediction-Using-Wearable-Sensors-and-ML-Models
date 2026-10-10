function eventTable = extractGaitEventsFromFile(filename)
%% =====================
% Load + Format CSV
%% =====================
raw = readtable(filename,'FileType','text','Delimiter',';', ...
'NumHeaderLines',6,'ReadVariableNames',false);
numRows = height(raw);
allData = cell(numRows,1);
maxCols = 0;
for r = 1:numRows
parts = strsplit(raw.Var1{r}, ',');
parts = strtrim(parts);
rowVals = NaN(1,numel(parts));
for k = 1:numel(parts)
if ~isempty(parts{k})
rowVals(k) = str2double(parts{k});
end
end
allData{r} = rowVals;
maxCols = max(maxCols, numel(rowVals));
end
data = NaN(numRows,maxCols);
for r = 1:numRows
data(r,1:numel(allData{r})) = allData{r};
end
%% =====================
% Gyro X (sagittal)
%% =====================
gx = -data(:,49);
gx = fillmissing(gx,'nearest');
fs = 148;
[b,a] = butter(2,6/(fs/2),'low');
gxF = filtfilt(b,a,gx);
t = (0:length(gxF)-1)'/fs;
absG = abs(gxF);
dw = [0; diff(gxF)];
%% =====================
% APA + Gait start
%% =====================
idxAPA = find(absG > 5,1,'first');
idxGait = find(absG > 20 & (1:length(absG))'>idxAPA,1,'first');
if isempty(idxGait)
eventTable = table();
return;
end
%% =====================
% Step detection (MSw)
%% =====================
gxGait = gxF(idxGait:end);
[~,locMSw_rel] = findpeaks(gxGait, ...
'MinPeakHeight',0.4*max(gxGait), ...
'MinPeakDistance',round(0.4*fs));
locMSw = locMSw_rel + idxGait - 1;
nSteps = numel(locMSw);
events = struct('TO1',[],'MSw',[],'HS1',[], ...
'TO2',[],'FF',[],'MSt',[],'HO',[]);
allEvents = repmat(events,nSteps,1);
%% =====================
% PASS 1: TO1, MSw, HS1, TO2
%% =====================
for s = 1:nSteps
iMSw = locMSw(s);
allEvents(s).MSw = iMSw;
if s==1
iPrev = idxGait;
else
iPrev = locMSw(s-1);
end
if s==nSteps
iNext = length(gxF);
else
iNext = locMSw(s+1);
end
% ---- TO1 (Eq.1)
if iMSw - iPrev >= 3
seg = -gxF(iPrev:iMSw);
[pks,locs] = findpeaks(seg);
if ~isempty(pks)
[~,id] = max(pks);
allEvents(s).TO1 = iPrev + locs(id) - 1;
end
end
% ---- HS1 (Eq.3)
if iNext - iMSw >= 3
seg = -gxF(iMSw:iNext);
[pks,locs] = findpeaks(seg);
if ~isempty(pks)
allEvents(s).HS1 = iMSw + locs(1) - 1;
end
end
% ---- TO2 (zero crossing)
iTO1 = allEvents(s).TO1;
if ~isempty(iTO1) && iMSw > iTO1+1
zseg = gxF(iTO1:iMSw);
zc = find(zseg(1:end-1).*zseg(2:end)<=0,1);
if ~isempty(zc)
allEvents(s).TO2 = iTO1 + zc - 1;
end
end
end
%% =====================
% PASS 2: FF, MSt, HO (HS1 → TO1(next))
%% =====================
epsFF = 10;
slopeEps = 2;
minDurSec = 0.05;
minSamples = max(3,round(minDurSec*fs));
for s = 1:nSteps-1
iHS1 = allEvents(s).HS1;
iTO1_next = allEvents(s+1).TO1;
if isempty(iHS1) || isempty(iTO1_next) || iTO1_next <= iHS1+3
continue;
end
win = iHS1:iTO1_next;
sig = gxF(win);
dws = dw(win);
%% ---- FF (Eq.5)
cond = (abs(sig) < epsFF) & (abs(dws) < slopeEps);
ffIdx = [];
if any(cond)
d = diff([0;cond;0]);
rs = find(d==1);
re = find(d==-1)-1;
len = re-rs+1;
good = find(len>=minSamples,1);
if ~isempty(good)
ffIdx = win(rs(good));
allEvents(s).FF = ffIdx;
end
end
%% ---- MSt (Eq.4, NEGATIVE region only)
negIdx = win(sig<0);
if ~isempty(negIdx)
[~,id] = min(abs(gxF(negIdx)));
mstIdx = negIdx(id);
if isempty(ffIdx) || mstIdx > ffIdx
allEvents(s).MSt = mstIdx;
end
end
%% ---- HO (Eq.6) AFTER MSt
iMSt = allEvents(s).MSt;
if isempty(iMSt)
continue;
end
hoWin = iMSt:iTO1_next;
hoSig = gxF(hoWin);
hoDw = dw(hoWin);
hoCond = (hoSig < -epsFF/2) & (hoDw > 0);
idh = find(hoCond,1);
if ~isempty(idh)
allEvents(s).HO = hoWin(idh);
end
end
%% =====================
% Build table (ordered)
%% =====================
rows = {};
order = {'TO1','MSw','HS1','TO2','FF','MSt','HO'};
for s = 1:nSteps
ev = allEvents(s);
for e = 1:numel(order)
idx = ev.(order{e});
if ~isempty(idx)
rows(end+1,:) = {s,order{e},idx,t(idx),gxF(idx)};
end
end
end
eventTable = cell2table(rows, ...
'VariableNames',{'Step','Event','Sample','Time_sec','Velocity_deg_per_s'});
end