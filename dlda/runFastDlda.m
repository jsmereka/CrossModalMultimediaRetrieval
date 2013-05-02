
load data


c=3;   % 3 classes
k=4;   % 3 topics. In this example, k=c, but we can also choose k>c

[M,V]=size(trainX);

initalpha=rand(k,1);
initbeta=[trainY,ones(M,1)-sum(trainY,2)]'*trainX;
initbeta=initbeta./(sum(initbeta,2)*ones(1,V));
lap=0.0001;
initeta=rand(k,c-1);

% if flag=1 use the change on perplexity to check the convergence, if flag=0, use the change on parameter to check the convergence
flag=1;

[alpha,beta,eta,phi,gama,logProb_time,perplexity_time]=learnFastDlda(trainX,trainY,initalpha,initbeta,initeta,lap,flag);

[predY,accuracy,perplexity,testphi,testgama]=applyFastDlda(testX,testY,alpha,beta,eta);