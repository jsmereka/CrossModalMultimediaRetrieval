function [predY,accuracy,perplexity,phi,gama]=applyFastDlda(X,Y,alpha,beta,eta)
%
%  Author: Hanhuai Shan. 04/2012
%
% k = number of topic
% c = number of classes
% N = number of words in a doc
% V = vocabulary size
% M = number of samples 
%
% Input:
%   X:      M*V, test docs
%   Y:      M*(c-1); M*(c-1), each row is the class label for one doc.  
%           The ith dimension with value 1 indicates the doc class is
%           i. If all dimensions are 0, the doc class is c.
%   alpha:  k*1, parameter for dirichlet distribution
%   beta:   k*V, parameter for discrete distribution
%   eta:    k*(c-1), parameter for regression for k-1 topics
%
% Output:
%   predY:      M*1; predicted labels on test set
%   accuracy:   scaler; classification accuracy
%   perplexity: scaler
%   phi:        M*k
%   gama:       M*k
%-----------------------------------------------------------------

disp(['Applying Fast DLDA...'])
M=size(X,1);
c=size(Y,2)+1;

% get phi on test documents
[phi,gama]=fastLdaEstep(alpha,beta,X);

% get perplexity on test documents
[logProb,perplexity]=fastDldaGetPerp(X,Y,alpha,beta,eta,phi,gama);

% get label on test documents
rawY=eta'*phi; %(c-1)*M
rawY=[rawY;zeros(1,M)];  %c*M
post=exp(rawY).*(ones(c,1)*(1./sum(exp(rawY),1)));
[C,predY]=max(post',[],2);

% get accuracy
[C,trueY]=max([Y,1-sum(Y,2)],[],2);
accuracy=sum(trueY==predY)/length(predY)



