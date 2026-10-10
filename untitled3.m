%% ===========================================
% PROCESS ALL SUBJECTS (MASTER LOOP)
% ===========================================
clear; clc;
% -------- Paths ---------
dataPath = 'C:\Users\user\Downloads\Data'; % Where S08, S09, ... are
savePath = 'C:\Users\user\Documents\Graduation Project/GaitEventsDetection';
% Create folder if it doesn't exist
if ~exist(savePath, 'dir')
mkdir(savePath);
end
% -------- Find subject folders --------
subjects = dir(fullfile(dataPath, 'S*')); % S08, S09, ...
for s = 1:length(subjects)
subjName = subjects(s).name; % e.g., 'S08'
subjFolder = fullfile(subjects(s).folder, subjName); % full path
fprintf('\n=============================\n');
fprintf(' Processing Subject %s\n', subjName);
fprintf('=============================\n');
% --- List ONLY walking CSV files ---
files = dir(fullfile(subjFolder, 'walk*.csv'));
allEventsSubject = []; % table accumulator
for f = 1:length(files)
filePath = fullfile(subjFolder, files(f).name);
fprintf(' → Processing %s\n', files(f).name);
% Run gait event extraction on this file
eventTbl = extractGaitEventsFromFile(filePath);
% Add filename column
eventTbl.FileName = repmat(string(files(f).name), height(eventTbl), 1);
% Append to subject table
allEventsSubject = [allEventsSubject; eventTbl];
end
% --- Save combined events for this subject ---
outFile = fullfile(savePath, sprintf('%s_GaitEvents.xlsx', subjName));
writetable(allEventsSubject, outFile);
fprintf(' ✔ Saved: %s\n', outFile);
end
fprintf('\n===== ALL SUBJECTS PROCESSED SUCCESSFULLY =====\n');