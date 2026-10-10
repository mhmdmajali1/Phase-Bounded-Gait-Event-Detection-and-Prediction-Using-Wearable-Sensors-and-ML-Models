clear all;
close all;
clc;
% This code train the KNN and use it to classify the different classes of
% the category

folders_path = strcat('.\CWTFeatures'); %features folders path
folders_path_results = strcat('.\Results\knn5\CWT');


num_classes = 4;
folder = dir(fullfile(folders_path, "/","S*"));
for subj = 1:length(folder)
    subj

    mkdir(strcat(folders_path_results,'/',folder(subj).name));%results directory
    %loading features and labels
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


    %knn model
    Mdl = fitcknn(Data.X,Data.y,'NumNeighbors',5,'KFold',10);
    ypred = kfoldPredict(Mdl);
    [cm,numcorrect,precision,recall,F] = getcm(Data.y(1:length(Data.X)),ypred,unique(labels)); 
    newname = strcat(folders_path_results, '/',folder(subj).name,'/');
    save(strcat(newname,'KNN_out'),'cm');
end