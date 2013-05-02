Code for Fast Discriminant Latent Dirichlet Allocation


The folder contains the matlab code for fast discriminant latent Dirichlet allocation (Fast DLDA). The model was proposed in the paper "H. Shan and A. Banerjee. Mixed-membership Naive Bayes Models. DMKD 2010"

The zip file contains the following files:

runFastDlda.m:                     An example on how to run the code.
learnFastDlda.m:                   Learn Fast DLDA from the training set. It calls fastDldaEstep.m and fastDldaMstep.m.
fastDldaEstep.m:                   Variational E-step.
fastDldaMstep.m:                   Variational M-step.
applyFastDlda.m:                   Apply Fast DLDA on the test set.
fastDldaGetPerp.m:                 Compute the perplexity
data.mat:                          Sample data (word counts).
readme.txt:                        Readme file