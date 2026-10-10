%% ===========================================
% PROCESS ALL SUBJECTS (MASTER LOOP)
% ===========================================
clear; clc;
% -------- Paths ---------
dataPath = 'C:\Users\user\Downloads\Data';
saveRoot = 'C:\Users\user\Documents\Graduation Project\GaitEventsDetection';

% Create root output folder
if ~exist(saveRoot, 'dir')
mkdir(saveRoot);
end


% -------- Find subject folders --------
subjects = dir(fullfile(dataPath, 'S*')); % S08, S09, ...
for s = 1:length(subjects)
subjName = subjects(s).name;
subjFolder = fullfile(subjects(s).folder, subjName);
fprintf('\n=============================\n');
fprintf(' Processing Subject %s\n', subjName);
fprintf('=============================\n');
% Create subject output folder
subjOutFolder = fullfile(saveRoot, subjName);
if ~exist(subjOutFolder, 'dir')
mkdir(subjOutFolder);
end

% Compute MVC reference for this subject (ONCE)
% --------------------------------------------------
fprintf(' → Computing MVC for %s\n', subjName);
%subjFolder = fullfile(rootData, subjName);
max_cont = preprocess_mvc(subjFolder, subjOutFolder);   % 8×1 vector

% --- List ONLY walking CSV files ---
files = dir(fullfile(subjFolder, 'walk*.csv'));
for f = 1:length(files)
filePath = fullfile(subjFolder, files(f).name);
[~, trialName, ~] = fileparts(files(f).name);

fprintf(' → Processing %s\n', files(f).name);

% Output folder for this walking trial
trialOutFolder = fullfile(subjOutFolder, trialName);
if ~exist(trialOutFolder, 'dir')
mkdir(trialOutFolder);
end

% Run IMU + EMG segmentation and saving
try
    extractIMUEMGEventsFromFile(filePath, trialOutFolder, max_cont);
catch ME
    fprintf('\n⚠ ERROR in %s\n', filePath);
    fprintf('Reason: %s\n', ME.message);
    fprintf('Skipping trial and continuing...\n');
end

end
end
fprintf('\n===== ALL SUBJECTS & TRIALS PROCESSED SUCCESSFULLY =====\n');