function [logProb,perplexity]=fastDldaGetPerp(X,Y,alpha,beta,eta,phi,gama)
%
%  Author: Hanhuai Shan. 04/2012
%
% k = number of topics
% c = number of classes
% N = number of words in a doc
% V = vocabulary size
% M = number of samples
% 
% Input:
%   X:      M*V; M docs, each is represented as word counts
%   Y:      M*(c-1), each row is the class label for one doc. The ith dimension with value 1
%           indicates the doc class is i. If all dimensions are 0, the doc class is c.
%   alpha:  k*1;
%   beta:   k*V;
%   eta:    k*(c-1);
%   phi:    k*M;
%   gama:   k*M;
%
% Ouptput:
%   logProb, perplexity
%---------------------------------------------------

[k,M]=size(phi);
[k,V]=size(beta);
Ns=sum(X,2);

item1=M*gammaln(sum(alpha))-M*sum(gammaln(alpha))+sum((alpha-1).*sum((psi(gama)-psi(ones(k,1)*sum(gama,1))),2));

item2=sum(sum(phi.*(psi(gama)-ones(k,1)*psi(sum(gama,1))),1).*Ns',2);

item4=sum(gammaln(sum(gama,1))-sum(gammaln(gama),1)+sum((gama-1).*(psi(gama)-ones(k,1)*psi(sum(gama,1)))));

item5=sum(sum(phi.*log(phi+realmin),1).*Ns');

item3=0;
for i=1:k
    temp=phi(i,:)'*log(beta(i,:)+realmin).*X;
    item3=item3+sum(sum(temp));
end

item6=0;
for d=1:M
    onephi=phi(:,d);
    oney=Y(d,:);
    item6=item6+onephi'*eta*oney'-log(1+sum(sum(exp(eta),2).*onephi));
end

logProb=item1+item2+item3-item4-item5+item6;
num=sum(sum(X));
perplexity=exp(-logProb/(num));
