%%%%%%%%%%%%%%%%%%%%
% demo2_matlab.m
% This demo shows how to do five-cross validation experiment 
% using sBP algorithm.
% This can be only used for Matlab because 'cvpartition' is a Matlab
% function.
%%%%%%%%%%%%%%%%%%%%

%%% Parameters
ALPHA = 1e-2;
BETA = 1e-2;
N = 500;
SEED = 1;
OUTPUT = 1;
J = 10;

%%% load data
load datasets/cora_wd
load datasets/cora_voc

%%% five-fold cross-validation

CVO = cvpartition(2410, 'kfold', 5);

for i = 1:CVO.NumTestSets
    fprintf(1, '\n*********************\nCross Validation %d\n*********************\n', i);
    trIdx = CVO.training(i);
    teIdx = CVO.test(i);
    fprintf(1, '\n*********************\nsBP train\n*********************\n');
    tic
    [phi, ~] = sBPtrain(cora_wd(:, trIdx), J, N, ALPHA, BETA, SEED, OUTPUT);
    toc
    tic
    fprintf(1, '\n*********************\nsBP predict\n*********************\n');
    theta = sBPpredict(cora_wd(:, teIdx), phi, N, ALPHA, BETA, SEED, OUTPUT);
    toc
end

