clear; clc; close all;
disp('SCRIPT STARTED');
drawnow;

%% ================================
% SETTINGS
%% ================================
rootFolder = 'C:\Users\user\Documents\Graduation Project\GaitEventsDetection';

eventNames = {'TO1','MSw','HS1'};
numClasses = numel(eventNames);

features_all = [];
labels_all   = [];
subject_id   = [];

%% ================================
% LOAD DATA SUBJECT BY SUBJECT
%% ================================
subjects = dir(fullfile(rootFolder,'S*'));
subjects = subjects([subjects.isdir]);

% sort folders numerically (S08, S09, S10...)
[~,idx] = sort(str2double(erase({subjects.name},'S')));
subjects = subjects(idx);

totalSubjects = length(subjects);
fprintf('Total subjects found: %d\n', totalSubjects);

for s = 1:length(subjects)

    subjPath = fullfile(subjects(s).folder, subjects(s).name);
    walks = dir(fullfile(subjPath,'walk*'));

    for w = 1:length(walks)

        walkPath = fullfile(walks(w).folder, walks(w).name);

        for e = 1:numClasses

            eventName = eventNames{e};
            filePath = fullfile(walkPath, [eventName '.xlsx']);

            if ~isfile(filePath)
                continue
            end

            T = readtable(filePath);

            validRows = ~isnan(T.TA_RMS);
            T = T(validRows,:);

            if isempty(T)
                continue
            end

            featMask = contains(T.Properties.VariableNames,...
                {'_RMS','_IEMG','_MAV','_WL','_ZC','_SSC','_PF','_TP'});

            F = T{:,featMask};

            features_all = [features_all; F];
            labels_all   = [labels_all; repmat(e,size(F,1),1)];
            subject_id   = [subject_id; repmat(s,size(F,1),1)];
        end
    end
end

fprintf('Total samples: %d\n', size(features_all,1));

%% ================================
% SUBJECT-WISE SPLIT
%% ================================
trainSubjects = 1:55;
testSubjects  = 56:length(subjects);

trainMask = ismember(subject_id, trainSubjects);
testMask  = ismember(subject_id, testSubjects);

X_train = features_all(trainMask,:);
y_train = labels_all(trainMask);

X_test  = features_all(testMask,:);
y_test  = labels_all(testMask);

fprintf('Training samples: %d\n', size(X_train,1));
fprintf('Testing samples : %d\n', size(X_test,1));

%% ================================
% NORMALIZATION (TRAIN STATS ONLY)
%% ================================
minX = min(X_train);
maxX = max(X_train);

X_train = (X_train - minX) ./ (maxX - minX + eps);
X_test  = (X_test  - minX) ./ (maxX - minX + eps);

%% ================================
% TRAIN RANDOM FOREST
%% ================================
RF = fitcensemble( ...
    X_train, y_train, ...
    'Method','Bag', ...
    'NumLearningCycles',50, ...
    'Learners','tree');

%% ================================
% TEST ON UNSEEN SUBJECTS
%% ================================
y_pred = predict(RF, X_test);

%% ================================
% RESULTS
%% ================================
figure;
confusionchart(y_test, y_pred, 'Normalization','row-normalized');
title('Subject-wise Random Forest Classification');

accuracy = sum(y_pred==y_test)/numel(y_test)*100;
fprintf('TEST Accuracy (unseen subjects) = %.2f %%\n', accuracy);

%% ================================
% PRECISION • RECALL • F1 SCORE
%% ================================
cm = confusionmat(y_test, y_pred);

precision = diag(cm) ./ sum(cm,1)';
recall    = diag(cm) ./ sum(cm,2);
f1        = 2 * (precision .* recall) ./ (precision + recall);

fprintf('\nClass-wise metrics:\n');

for i = 1:numClasses
    fprintf('%s:\n', eventNames{i});
    fprintf(' Precision: %.3f\n', precision(i));
    fprintf(' Recall:    %.3f\n', recall(i));
    fprintf(' F1-score:  %.3f\n\n', f1(i));
end

fprintf('Macro F1-score: %.3f\n', mean(f1));

%% ================================
% FEATURE IMPORTANCE
%% ================================
importance = predictorImportance(RF);

figure;
bar(importance);
title('Feature Importance');
xlabel('Feature Index');
ylabel('Importance');