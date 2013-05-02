function [Y1,post1,post2,perplexity1,perplexity2,phi,gama]=predictSlda2(X,Y,alpha,beta,eta)
%
% k = number of classes
% N = number of words in a doc
% V = vocabulary size
% M = number of samples 
%
% Input:
%   X:      M*V, test docs
%   alpha:  k*1, parameter for dirichlet distribution
%   beta:   k*V, parameter for discrete distribution
%   eta:    k*(k-1), parameter for regression for k-1 classes
%
% Output:
%   Y1:      M*1, the predicted labels for M test docs
%-----------------------------------------------------------------



phi=[];
gama=[];
M=size(X,1);
k=length(alpha);
c=size(Y,2)+1;


    

% for d=1:M
%     x_d=X(d,:);
%     [estimatedPhi,estimatedGamma]=lda2Estep(alpha,beta,x_d);
%     phi(:,d)=estimatedPhi;
%     gama(:,d)=estimatedGamma;      
% end

[phi,gama]=lda2Estep(alpha,beta,X);
[logProb1,perplexity1]=applySlda2(X,Y,alpha,beta,eta,phi,gama);
[logProb2,perplexity2]=applyLda2(X,alpha,beta,phi,gama);

% phi=permute(phi,[1,3,2]);   %phi: k*M*N
% temp=repmat(full(X),[1,1,k]);
% temp=permute(temp,[3,1,2]); %temp: k*M*N
% avgphi=sum(phi.*temp,3)./(ones(k,1)*sum(X,2)');   %avgphi: k*M, the
% average of phi over N words for M docs.



rawY=eta'*phi; %(c-1)*M
rawY=[rawY;zeros(1,M)];  %c*M
post1=exp(rawY).*(ones(c,1)*(1./sum(exp(rawY),1)));
[C,Y1]=max(post1',[],2);

if k==length(Y)
    post2=gama./(ones(k,1)*sum(gama,1));
else
    post2=0;
end

