%%%%%%%%%%%%%%%%%%%%%%%%
% This quickstart.m confirms all functions working properly.
% It also shows how to use each function in the toolbox.
%%%%%%%%%%%%%%%%%%%%%%%%

%%% Parameters
ALPHA = 1e-2;
BETA = 1e-2;
OMEGA = 0.05;
LAMBDA = 0.5;
TD = 0.2;
TW = 0.2;
TK = 0.5;
N = 20;
M = 1;
SEED = 1;
OUTPUT = 1;
J = 10;

%%% load data
fprintf(1, 'CORA\n*********************\n');
load datasets/cora_wd
load datasets/cora_ad
load datasets/cora_dd

%%% Gibbs Sampling algorithm (GS)
fprintf(1, 'GS\n*********************\n');
tic
[phi, theta, z] = GStrain(cora_wd, J, N, ALPHA, BETA, SEED, OUTPUT);
[theta, z] = GSpredict(cora_wd, phi, N, ALPHA, BETA, SEED, OUTPUT);
toc
fprintf(1, '\n*********************\n');

%%% Variational Bayes algorithm (VB)
fprintf(1, 'VB\n*********************\n');
tic
[phi, theta, mu] = VBtrain(cora_wd, J, N, M, ALPHA, BETA, SEED, OUTPUT); 
[theta, mu] = VBpredict(cora_wd, phi, N, M, ALPHA, BETA, SEED, OUTPUT); 
toc
fprintf(1, '\n*********************\n');

%%% Variational Bayes algorithm (VB) (pure matlab codes)
% fprintf(1, 'LDAVB (pure matlab codes)\n*********************\n');
% tic
% [phi, theta] = LDAVBtrain(cora_wd, J, N, M, ALPHA, BETA, OUTPUT); 
% theta = LDAVBpredict(cora_wd, phi, N, M, ALPHA, BETA, OUTPUT); 
% toc
% fprintf(1, '\n*********************\n');

%%% Synchronous Belief Propagation algorithm (sBP)
fprintf(1, 'sBP\n*********************\n');
tic
[phi, theta, mu] = sBPtrain(cora_wd, J, N, ALPHA, BETA, SEED, OUTPUT);
[theta, mu] = sBPpredict(cora_wd, phi, N, ALPHA, BETA, SEED, OUTPUT);
toc
fprintf(1, '\n*********************\n');

%%% Asynchronous Belief Propagation algorithm (aBP)
fprintf(1, 'aBP\n*********************\n');
tic
[phi, theta, mu] = aBPtrain(cora_wd, J, N, ALPHA, BETA, SEED, OUTPUT);
[theta, mu] = aBPpredict(cora_wd, phi, N, ALPHA, BETA, SEED, OUTPUT);
toc
fprintf(1, '\n*********************\n');

%%% Synchronous Collapsed Variational Bayes 0 algorithm (CVB0)
fprintf(1, 'sCVB0\n*********************\n');
tic
[phi, theta, mu] = sCVB0train(cora_wd, J, N, ALPHA, BETA, SEED, OUTPUT);
[theta, mu] = sCVB0predict(cora_wd, phi, N, ALPHA, BETA, SEED, OUTPUT);
toc
fprintf(1, '\n*********************\n');

%%% Asynchronous Collapsed Variational Bayes 0 algorithm (CVB0)
fprintf(1, 'aCVB0\n*********************\n');
tic
[phi, theta, mu] = aCVB0train(cora_wd, J, N, ALPHA, BETA, SEED, OUTPUT);
[theta, mu] = aCVB0predict(cora_wd, phi, N, ALPHA, BETA, SEED, OUTPUT);
toc
fprintf(1, '\n*********************\n');

%%% Synchronous Simplified Belief Propagation algorithm (ssiBP)
fprintf(1, 'ssiBP\n*********************\n');
tic
[phi, theta, mu] = ssiBPtrain(cora_wd, J, N, ALPHA, BETA, SEED, OUTPUT);
[theta, mu] = ssiBPpredict(cora_wd, phi, N, ALPHA, BETA, SEED, OUTPUT);
toc
fprintf(1, '\n*********************\n');

%%% Asynchronous Simplified Belief Propagation algorithm (asiBP)
fprintf(1, 'asiBP\n*********************\n');
tic
[phi, theta, mu] = asiBPtrain(cora_wd, J, N, ALPHA, BETA, SEED, OUTPUT);
[theta, mu] = asiBPpredict(cora_wd, phi, N, ALPHA, BETA, SEED, OUTPUT);
toc
fprintf(1, '\n*********************\n');

%%% Synchronous Simplified Belief Propagation algorithm (LDAsiBP) (pure matlab codes)
fprintf(1, 'LDAsiBP (pure matlab codes)\n*********************\n');
tic
[phi, theta] = LDAssiBPtrain(cora_wd, J, N, ALPHA, BETA, OUTPUT);
theta = LDAssiBPpredict(cora_wd, phi, N, ALPHA, BETA, OUTPUT);
toc
fprintf(1, '\n*********************\n');

%%% Gibbs Sampling for Author-Topic Models (ATMGS)
fprintf(1, 'ATMGS\n*********************\n');
tic
[phi, theta, z, x] = ATMGStrain(cora_wd, cora_ad, J, N, ALPHA, BETA, SEED, OUTPUT);
[theta, z, x] = ATMGSpredict(cora_wd, cora_ad, phi, N, ALPHA, BETA, SEED, OUTPUT);
toc
fprintf(1, '\n*********************\n');

%%% Synchronous Belief Propagation for Author-Topic Models (ATMBP)
fprintf(1, 'ATMBP\n*********************\n');
tic
[phi, theta, mu, x] = ATMBPtrain(cora_wd, cora_ad, J, N, ALPHA, BETA, SEED, OUTPUT);
[theta, mu, x] = ATMBPpredict(cora_wd, cora_ad, phi, N, ALPHA, BETA, SEED, OUTPUT);
toc
fprintf(1, '\n*********************\n');

%%% Synchronous Belief Propagation for Relational Topic Models (RTMBP)
fprintf(1, 'RTMBP\n*********************\n');
tic
[phi, theta, gamma, mu] = RTMBPtrain(cora_wd, cora_dd, J, N, OMEGA, ALPHA, BETA, SEED, OUTPUT);
[theta, mu] = RTMBPpredict(cora_wd, cora_dd, phi, gamma, N, OMEGA, ALPHA, BETA, SEED, OUTPUT);
toc
fprintf(1, '\n*********************\n');

%%% Synchronous Belief Propagation for Labeled Latent Dirichlet Allocation
%%% (LaLDABP)
fprintf(1, 'LaLDABP\n*********************\n');
tic
[phi, theta, z] = LaLDABPtrain(cora_wd, cora_ad, N, ALPHA, BETA, SEED, OUTPUT);
toc
fprintf(1, '\n*********************\n');

%%% Residual Belief Propagation for Latent Dirichlet Allocation (RBP)
fprintf(1, 'RBP_doc\n*********************\n');
tic
[phi, theta, mu] = RBPtrain_doc(cora_wd, J, N, ALPHA, BETA, SEED, OUTPUT);
[theta, mu] = RBPpredict_doc(cora_wd, phi, N, ALPHA, BETA, SEED, OUTPUT);
toc
fprintf(1, '\n*********************\n');

fprintf(1, 'RBP_voc\n*********************\n');
tic
[phi, theta, mu] = RBPtrain_voc(cora_wd', J, N, ALPHA, BETA, SEED, OUTPUT);
[theta, mu] = RBPpredict_voc(cora_wd', phi, N, ALPHA, BETA, SEED, OUTPUT);
toc
fprintf(1, '\n*********************\n');

%%% Fast Gibbs Sampling for Latent Dirichlet allocation (FGS)
fprintf(1, 'FGS\n*********************\n');
tic
[phi, theta] = FGStrain(cora_wd, J, N, ALPHA, BETA, SEED, OUTPUT);
theta = FGSpredict(cora_wd, phi, N, ALPHA, BETA, SEED, OUTPUT);
toc

%%% Fast Belief Propagation for Latent Dirichlet Allocation (FBP)
fprintf(1, 'FBP_doc\n*********************\n');
tic
[phi, theta, mu] = FBPtrain_doc(cora_wd, J, LAMBDA, N, ALPHA, BETA, SEED, OUTPUT);
[theta, mu] = FBPpredict_doc(cora_wd, phi, LAMBDA, N, ALPHA, BETA, SEED, OUTPUT);
toc

fprintf(1, 'FBP_voc\n*********************\n');
tic
[phi, theta, mu] = FBPtrain_voc(cora_wd', J, LAMBDA, N, ALPHA, BETA, SEED, OUTPUT);
[theta, mu] = FBPpredict_voc(cora_wd', phi, LAMBDA, N, ALPHA, BETA, SEED, OUTPUT);
toc
fprintf(1, '\n*********************\n');

%%% Active Belief Propagation for Latent Dirichlet Allocation (ABP)
fprintf(1, 'ABP_doc\n*********************\n');
tic
[phi, theta, mu] = ABPtrain_doc(cora_wd, J, TD, TK, N, ALPHA, BETA, SEED, OUTPUT);
[theta, mu] = ABPpredict_doc(cora_wd, phi, TD, TK, N, ALPHA, BETA, SEED, OUTPUT);
toc

fprintf(1, 'ABP_voc\n*********************\n');
tic
[phi, theta, mu] = ABPtrain_voc(cora_wd', J, TW, TK, N, ALPHA, BETA, SEED, OUTPUT);
[theta, mu] = ABPpredict_voc(cora_wd', phi, TW, TK, N, ALPHA, BETA, SEED, OUTPUT);
toc

%%% Tiny Belief Propagation for Latent Dirichlet Allocation (TBP)
%%% Synchronous TBP
fprintf(1, 'sTBP\n*********************\n');
tic
[phi, theta] = sTBPtrain(cora_wd, J, N, ALPHA, BETA, SEED, OUTPUT);
[theta] = sTBPpredict(cora_wd, phi, N, ALPHA, BETA, SEED, OUTPUT);
toc
fprintf(1, '\n*********************\n');

%%% Asynchronous TBP
fprintf(1, 'aTBP\n*********************\n');
tic
[phi, theta] = aTBPtrain(cora_wd, J, N, ALPHA, BETA, SEED, OUTPUT);
[theta] = aTBPpredict(cora_wd, phi, N, ALPHA, BETA, SEED, OUTPUT);
toc
fprintf(1, '\n*********************\n');

%%% Synchronous TBP for file
fprintf(1, 'sTBP for file\n*********************\n');
tic
[phi, theta] = sTBPtrain_file('datasets/kos_mat.txt', J, N, ALPHA, BETA, SEED, OUTPUT);
[theta] = sTBPpredict_file('datasets/kos_mat.txt', phi, N, ALPHA, BETA, SEED, OUTPUT);
toc
fprintf(1, '\n*********************\n');

%%% Synchronous TBP for file
fprintf(1, 'aTBP for file\n*********************\n');
tic
[phi, theta] = aTBPtrain_file('datasets/kos_mat.txt', J, N, ALPHA, BETA, SEED, OUTPUT);
[theta] = aTBPpredict_file('datasets/kos_mat.txt', phi, N, ALPHA, BETA, SEED, OUTPUT);
toc
fprintf(1, '\n*********************\n');

%%% Utilities MAT2WD and UCI2WD
fprintf(1, 'Transform MAT to WD format\n*********************\n');
wd = MAT2WD('datasets/kos_mat.txt');
fprintf(1, '\n*********************\n');

fprintf(1, 'Transform UCI to WD format\n*********************\n');
wd = UCI2WD('datasets/kos_uci.txt');
fprintf(1, '\n*********************\n');

fprintf(1, '\n*********************\n\n*********************\n All functions work properly! \n*********************\n\n*********************\n');