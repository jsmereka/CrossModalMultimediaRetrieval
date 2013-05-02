function [phi_t,gama_t]=fastDldaEstep(alpha,beta,eta,phi_last,X,Y)
%
%  Author: Hanhuai Shan. 04/2012
% 
% k = number of topics
% c = number of classes
% N = number of words in a doc
% V = vocabulary size
% M = number of documents
%
% Input:
%   alpha:      k*1, parameter of Dirichlet distribution
%   beta:       k*V, paramters for k topics to generate the words
%   eta:        k*(c-1), regression parameter for c-1 classes, if
%               #class=#cluster, c=k;
%   phi_last:   k*M, estimation of phi from the last iteration
%   X:          M*V, M documents, a doc represented by words' occurence (e.g.[3,0,2..])
%   Y:          M*(c-1), each row is the class label for one doc. The ith dimension with value 1
%               indicates the doc class is i. If all dimensions are 0, the doc class is c.
%
% output:
%   phi_t:      k*M
%   gama_t:     k*M 
% -------------------------------------------------------------------


[k,V]=size(beta);
M=size(X,1);
Ns=sum(X,2); 


phi_t=ones(k,M)/k;
gama_t=alpha*ones(1,M)+ones(k,1)*Ns'/k;

epsilon=0.001;
time=500;

e=100;
t=1;

xis=1+sum(sum(exp(eta),2)*ones(1,M).*phi_last,1)';

while e>epsilon && t<time
    % update phi
    temp=gama_t-ones(k,1)*sum(gama_t,1)+(X*log(beta'+realmin)./(Ns*ones(1,k)))'+eta*Y'./(ones(k,1)*Ns')-sum(exp(eta),2)*(1./(Ns'.*xis'));
    maxtemp=max(temp,[],1);
    phi_tt=exp(temp+ones(k,1)*abs(maxtemp));
    
    phi_tt=phi_tt./(ones(k,1)*sum(phi_tt,1));
    
    % update gamma
    gama_tt=alpha*ones(1,M)+phi_tt.*(ones(k,1)*Ns');
   
    % difference from the previous update
    e1=sum(sum(abs(phi_tt-phi_t)))/sum(sum(phi_t));
    e2=sum(sum(abs(gama_tt-gama_t)))/sum(sum(gama_t));
    e=max(e1,e2);
    
    phi_t=phi_tt;
    gama_t=gama_tt;
%     disp(['t=',int2str(t),', e1,e2,e:',num2str(e1),',',num2str(e2),',',num2str(e)]);
    t=t+1;
end

