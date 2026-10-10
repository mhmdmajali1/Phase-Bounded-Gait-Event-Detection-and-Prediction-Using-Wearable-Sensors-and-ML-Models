clear all;
close all;
clc;

% This code train the SVM and use it to classify the different classes of the category

folders_path = strcat('.\feature extraction\CWTFeatures'); %features folders path
folders_path_results = strcat('.\Results\SVM\CWT');

num_classes = 4;
folder = dir(fullfile(folders_path, "/","S*"));

for subj = 1:length(folder)
    
    mkdir(strcat(folders_path_results,'/',folder(subj).name));
    file1 = dir(fullfile(folders_path,'/',folder(subj).name,'/','*tures.mat'));
    file2 = dir(fullfile(folders_path,'/',folder(subj).name,'/','*abels.mat'));
    Features_path = strcat(folders_path, '/',folder(subj).name,'/',  file1(1).name);
    Labels_path = strcat(folders_path, '/',folder(subj).name,'/',  file2(1).name);
    load(Features_path);
    load(Labels_path);
    
    features=abs(features);
    [t, m]=size(features);
    
    Data.X = features';
    Data.y = labels';
    %min-max Normalization :
    Data.X = (Data.X - repmat(min(Data.X ,[],1),size(Data.X,1),1))*spdiags(1./(max(Data.X,[],1)-min(Data.X,[],1))',0,size(Data.X,2),size(Data.X,2));
    
    
    options.arg = [0.01, 0.1, 1,  5 ];%2^-9 - 2^3
    % options.arg = [0.0001, 0.01, 0.1, 0.5, 1, 1.5, 5, 10];
    options.C = [ 0.5, 1, 5, 10];
    options.num_folds = 10;
    options.verb = 1;
    
    for cv=1:1
        %             [model,Errors,best_modelParameters,tst] = hyperparameterSVM(Data,options);
        %             labels = tst.y;
        %             labels = labels';
        %             samples = tst.X;
        %             samples = samples';
        
        
        [best_model,Errors,best_modelParameters] = hyperparameterSVM(Data,options);
        
        ypred= kfoldPredict(best_model);
        [cm,numcorrect,precision,recall,F] = getcm (labels,ypred,unique(labels));
        SVM_out (cv) = struct('model',best_model,'F',F,'precision',precision,'recall',recall,'numcorrect',numcorrect,'cm',cm,'ypred',ypred,'accuracy',accuracy);
    end
    %     SVM_out = struct('cm',cm);
    
    newname = strcat(folders_path_results, '\',folder(subj).name,'\');
    save(strcat(newname,'SVM_out'),'SVM_out');
    
    clear SVM_out;
    clear labels;
    clear samples;
end