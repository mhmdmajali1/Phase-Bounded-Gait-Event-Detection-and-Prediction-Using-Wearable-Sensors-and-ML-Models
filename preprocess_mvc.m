function max_cont = preprocess_mvc(subjectFolder, subjectOutFolder)
% -------------------------------------------------------------
% preprocess_mvc
% -------------------------------------------------------------
% Loads MVC files for one subject, preprocesses EMG,
% and computes MVC reference per muscle.
%
% OUTPUT:
%   max_cont : 8x1 vector (MVC per muscle)
%   Also SAVES an Excel file inside subject folder
% -------------------------------------------------------------

%% ------------------------------------------------------------
% Configuration
%% ------------------------------------------------------------
fsTarget = 1778;

muscleNames = {'TA','GM','GL','RF','VL','VM','ST','BF'};

% original EMG sampling rates per channel
fsOrig = [1259 1778 1778 1259 1778 1778 1259 1259];

mvcFiles = { ...
    'mvc_hamstring_1', ...
    'mvc_quad_1', ...
    'mvc_shank_1' ...
};

%% ------------------------------------------------------------
% Find MVC files
%% ------------------------------------------------------------
mvcPaths = [];

for i = 1:numel(mvcFiles)
    f = dir(fullfile(subjectFolder, [mvcFiles{i} '*.csv']));
    for k = 1:numel(f)
        mvcPaths{end+1} = fullfile(f(k).folder, f(k).name);
    end
end

if isempty(mvcPaths)
    error('No MVC files found in subject folder');
end

%% ------------------------------------------------------------
% Store max per file (8 x Nfiles)
%% ------------------------------------------------------------
mvcMaxAll = [];

%% ------------------------------------------------------------
% Process each MVC file
%% ------------------------------------------------------------
for f = 1:numel(mvcPaths)

    %% Load CSV (same loader you use elsewhere)
    raw = readtable(mvcPaths{f}, 'FileType','text','Delimiter',';', ...
        'NumHeaderLines',6,'ReadVariableNames',false);

    numRows = height(raw);
    allData = cell(numRows,1);
    maxCols = 0;

    for r = 1:numRows
        parts = strsplit(raw.Var1{r}, ',');
        rowVals = NaN(1,numel(parts));
        for k = 1:numel(parts)
            if ~isempty(parts{k})
                rowVals(k) = str2double(parts{k});
            end
        end
        allData{r} = rowVals;
        maxCols = max(maxCols, numel(rowVals));
    end

    data = NaN(numRows, maxCols);
    for r = 1:numRows
        data(r,1:numel(allData{r})) = allData{r};
    end

    %% Extract raw EMG channels
    rawEMG = { ...
        data(:,1),  data(:,8),  data(:,9),  data(:,16), ...
        data(:,23), data(:,24), data(:,31), data(:,38) ...
    };

    EMGproc = zeros(0,8);

    %% Preprocess each channel
    for m = 1:8
        x = rawEMG{m};

        % 1. Replace NaN / Inf
        x(~isfinite(x)) = 0;

        % 2. Resample
        x = resample(x, fsTarget, fsOrig(m));

        % 3. Band-pass
        x = emgBandpass(x, fsTarget);

        % 4. Notch
        x = emgNotch(x, fsTarget);

        % 5. Rectify
        x = abs(x);

        % store
        EMGproc(1:length(x),m) = x;
    end

    %% MVC per channel for this file
    mvcMaxAll(:,end+1) = max(EMGproc, [], 1)';

end

%% ------------------------------------------------------------
% Final MVC per muscle (max across MVC files)
%% ------------------------------------------------------------
max_cont = max(mvcMaxAll, [], 2);

%% ------------------------------------------------------------
% Save MVC reference file
%% ------------------------------------------------------------
row1 = [{'Muscle'}, muscleNames];              % header row
row2 = [{'MVC_Value'}, num2cell(max_cont')];   % values row

T = cell2table([row1; row2]);

outFile = fullfile(subjectOutFolder, 'MVC_reference.xlsx');
if exist(outFile,'file')
    delete(outFile);
end
writetable(T, outFile, 'WriteVariableNames', false);

fprintf('MVC reference saved: %s\n', outFile);

end

function y = emgBandpass(x, fs)
    [b,a] = butter(4, [20 450]/(fs/2), 'bandpass');
    y = filtfilt(b,a,x);
end

function y = emgNotch(x, fs)
    W0 = 50/(fs/2);
    BW = W0/35;
    [b,a] = iirnotch(W0, BW);
    y = filtfilt(b,a,x);
end