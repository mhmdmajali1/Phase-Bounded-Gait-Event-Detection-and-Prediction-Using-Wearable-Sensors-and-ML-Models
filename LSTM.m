clear; clc; close all;
disp('LSTM SCRIPT STARTED');

%% ================================
% SETTINGS
%% ================================
rootFolder = 'C:\Users\user\Documents\Graduation Project\GaitEventsDetection';

eventNames = {'TO','MSw','HS'};
numClasses = numel(eventNames);

subjects = dir(fullfile(rootFolder,'S*'));
subjects = subjects([subjects.isdir]);
[~,idx] = sort({subjects.name});
subjects = subjects(idx);
totalSubjects = length(subjects);
disp(totalSubjects)

rng(1)  % for reproducibility

idx = randperm(totalSubjects);
trainIdx = idx(1:55);
testIdx  = idx(56:end);

XTrain = {};
YTrain = {};

XTest = {};
YTest = {};

%% ================================
% LOOP THROUGH SUBJECTS
%% ================================
for s = 1:length(subjects)

    subjPath = fullfile(subjects(s).folder, subjects(s).name);
    walks = dir(fullfile(subjPath,'walk*'));

    for w = 1:length(walks)
        clear allData

        walkPath = fullfile(walks(w).folder, walks(w).name);

        allSeqFeatures = [];
        allSeqLabels   = [];

        for e = 1:numClasses

            filePath = fullfile(walkPath,[eventNames{e} '.xlsx']);
            if ~isfile(filePath)
                continue
            end

            T = readtable(filePath);

            validRows = find(~isnan(T.TA_RMS));
            if isempty(validRows)
                continue
            end

            featMask = contains(T.Properties.VariableNames,...
                {'_RMS','_IEMG','_MAV','_WL','_ZC','_SSC','_PF','_TP'});

            F = fillmissing(T{validRows, featMask}, 'constant', 0);

            if size(F,2) ~= 64
                error('Feature count mismatch in %s', filePath);
            end

            steps = T.Step(validRows);
            segments = T.Segment(validRows);

            tmp = table;
            tmp.Features = num2cell(F,2);
            tmp.Step = steps;
            tmp.Segment = segments;
            tmp.Label = repmat(e,length(validRows),1);

            if exist('allData','var')
                allData = [allData; tmp];
            else
                allData = tmp;
            end
        end

        if ~exist('allData','var') || isempty(allData)
            continue
        end

        % Sort chronologically
        allData = sortrows(allData,{'Step','Segment'});

        seqFeatures = cell2mat(allData.Features);
        seqLabels   = allData.Label;

        if size(seqFeatures,1) < 5
            clear allData
            continue
        end

        % Shift for next-event prediction
        window = 3;   % try 3 → 5

        Xseq = [];
        Yseq = [];

        for t = window : size(seqFeatures,1)-1
            segment = seqFeatures(t-window+1:t,:)';
            Xseq(:,end+1) = segment(:);   % flatten window
            Yseq(end+1) = seqLabels(t+1);
        end

        Yseq = categorical(Yseq);

        if size(Xseq,2) > 200
            continue
        end

        if ismember(s, trainIdx)
            XTrain{end+1} = Xseq;
            YTrain{end+1} = Yseq;
        elseif ismember(s, testIdx)
            XTest{end+1} = Xseq;
            YTest{end+1} = Yseq;
        end

        clear allData
    end
end

%% ================================
% NORMALIZATION (TRAIN ONLY)
%% ================================
allTrainData = cat(2,XTrain{:});
minX = min(allTrainData,[],2);
maxX = max(allTrainData,[],2);

for i=1:length(XTrain)
    XTrain{i} = (XTrain{i} - minX) ./ (maxX - minX + eps);
end

for i=1:length(XTest)
    XTest{i} = (XTest{i} - minX) ./ (maxX - minX + eps);
end

disp('Feature dimension check:')
disp(size(XTrain{1},1))
disp('Classes in training labels:')
disp(categories([YTrain{:}]))
lengths = cellfun(@(x) size(x,2), XTrain);
fprintf('Min length: %d\n', min(lengths));
fprintf('Max length: %d\n', max(lengths));
fprintf('Mean length: %.1f\n', mean(lengths));
disp('Class distribution in training data:')
countcats([YTrain{:}])
%% ================================
% DATA VALIDATION (PREVENT CRASH)
%% ================================
disp('===== DATA VALIDATION =====')

assert(~isempty(XTrain), '❌ XTrain is empty');
assert(~isempty(YTrain), '❌ YTrain is empty');

assert(numel(XTrain) == numel(YTrain), ...
    '❌ XTrain and YTrain size mismatch');

for i = 1:numel(XTrain)

    X = XTrain{i};
    Y = YTrain{i};

    % check numeric type
    assert(isfloat(X), '❌ X must be numeric');

    % check NaN
    assert(~any(isnan(X(:))), '❌ NaN values found in X');

    % check dimensions
    assert(size(X,2) == numel(Y), ...
        '❌ Sequence length mismatch between X and Y');

    % check row vector labels
    assert(isrow(Y), '❌ Y must be row vector');

    % check categorical
    assert(iscategorical(Y), '❌ Y must be categorical');

end

disp('✅ DATA VALIDATION PASSED')

%% ================================
% DEFINE LSTM NETWORK
%% ================================
numFeatures = size(XTrain{1},1);   % 8 muscles × 8 features
numHiddenUnits = 192;
numClasses = 3;     % TO1, MSw, HS1

layers = [
    sequenceInputLayer(numFeatures)
    lstmLayer(numHiddenUnits,'OutputMode','sequence')
    dropoutLayer(0.3)
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer];

options = trainingOptions('adam', ...
    'MaxEpochs',50,...
    'MiniBatchSize',32,...
    'InitialLearnRate',0.001,...
    'LearnRateSchedule','piecewise',...
    'LearnRateDropFactor',0.3,...
    'LearnRateDropPeriod',15,...
    'Shuffle','every-epoch',...
    'Verbose',true,...
    'Plots','training-progress');

size(XTrain{1})
size(YTrain{1})
class(YTrain{1})

disp('Number of training sequences:')
disp(length(XTrain))
disp('Number of training label sequences:')
disp(length(YTrain))

disp('Number of test sequences:')
disp(length(XTest))
disp('Number of test label sequences:')
disp(length(YTest))

%% ================================
% TRAIN NETWORK
%% ================================
net = trainNetwork(XTrain,YTrain,layers,options);

%% ================================
% TEST NETWORK
%% ================================
YPred = classify(net,XTest,'MiniBatchSize',1);

%% ================================
% EVALUATION
%% ================================
totalCorrect = 0;
totalCount = 0;

for i = 1:numel(YPred)
    totalCorrect = totalCorrect + sum(YPred{i} == YTest{i});
    totalCount   = totalCount + numel(YTest{i});
end

accuracy = totalCorrect / totalCount * 100;

fprintf('LSTM Test Accuracy = %.2f %%\n',accuracy);

trueLabels = [];
predLabels = [];

for i = 1:numel(YTest)
    trueLabels = [trueLabels YTest{i}];
    predLabels = [predLabels YPred{i}];
end

figure;
confusionchart(trueLabels, predLabels, 'Normalization','row-normalized');
title('LSTM Next Event Prediction');

%% ================================
% PRECISION • RECALL • F1 SCORE
%% ================================
classes = categories(trueLabels);
cm = confusionmat(trueLabels, predLabels);

precision = diag(cm) ./ sum(cm,1)';
recall    = diag(cm) ./ sum(cm,2);
f1        = 2 * (precision .* recall) ./ (precision + recall);

fprintf('\nClass-wise metrics:\n');

for i = 1:length(classes)
    fprintf('%s:\n', classes{i});
    fprintf(' Precision: %.3f\n', precision(i));
    fprintf(' Recall:    %.3f\n', recall(i));
    fprintf(' F1-score:  %.3f\n\n', f1(i));
end

fprintf('Macro F1-score: %.3f\n', mean(f1));