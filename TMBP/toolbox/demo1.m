%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This demo1.m runs Synchronous Belief Propagation (sBP) for LDA.
% The results are printed on the screen including
% training perplexity and five top words in each topic.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%% Parameters
ALPHA = 1e-2;
BETA = 1e-2;
N = 500;
M = 1;
SEED = 1;
OUTPUT = 1;
J = 10;

%%% load data
load datasets/cora_wd
load datasets/cora_voc

%%% The Belief Propagation algorithm
fprintf(1, '\n*********************\nThe sBP Algorithm\n*********************\n');
tic
[phi, theta, mu] = sBPtrain(cora_wd, J, N, ALPHA, BETA, SEED, OUTPUT);
toc
fprintf(1, '\n*********************\nTop five words in each of ten topics by sBP\n*********************\n');
topicshow(phi, cora_voc, 5);