%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This test.m compares the three learning
% algorithms for LDA such as Gibbs sampling (GS),
% Variational Bayes (VB) and Belief Propagation (BP).
% The results are printed on the screen including 
% the training perplexity and five top words of each topic.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%% Parameters
ALPHA = 1e-2;
BETA = 1e-2;
N = 1000;
M = 1;
SEED = 1;
OUTPUT = 1;
J = 10;

%%% Load data
load datasets/cora_wd
load datasets/cora_voc

%%% The Gibbs sampling algorithm
fprintf(1, '\n*********************\nThe GS Algorithm\n*********************\n');
tic
[phi, theta, z] = GStrain(cora_wd, J, N, ALPHA, BETA, SEED, OUTPUT);
toc
fprintf(1, '\n*********************\nTop five words in each of ten topics by GS\n*********************\n');
topicshow(phi, cora_voc, 5);

%%% The Variational Bayes algorithm
fprintf(1, '\n*********************\nThe VB Algorithm\n*********************\n');
tic
[phi, theta, mu] = VBtrain(cora_wd, J, N, M, ALPHA, BETA, SEED, OUTPUT);  
toc
fprintf(1, '\n*********************\nTop five words in each of ten topics by VB\n*********************\n');
topicshow(phi, cora_voc, 5);

%%% The Belief Propagation algorithm
fprintf(1, '\n*********************\nThe sBP Algorithm\n*********************\n');
tic
[phi, theta, mu] = sBPtrain(cora_wd, J, N, ALPHA, BETA, SEED, OUTPUT);
toc
fprintf(1, '\n*********************\nTop five words in each of ten topics by sBP\n*********************\n');
topicshow(phi, cora_voc, 5);
