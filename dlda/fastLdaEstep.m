function [phi_t,gama_t]=fastLdaEstep(alpha,beta,X)
%
%  Author: Hanhuai Shan. 04/2012
% 
% k = number of topics
% N = number of words in a doc
% V = vocabulary size
% M = number of samples
%
% Input:
%   alpha:      k*1, parameter of Dirichlet distribution
%   beta:       k*V, paramters for k discrete distributions
%   X:          M*V, a doc represented by words' occurence (e.g.[3,0,2..])
%
% output:
%   phi_t:      k*M matrix
%   gama_t:     k*M matrix
% -------------------------------------------------------------------


[k,V]=size(beta);
Ns=sum(X,2); 
M=size(X,1);


phi_t=ones(k,M)/k;
gama_t=alpha*ones(1,M)+ones(k,1)*Ns'/k;


epsilon=0.001;
time=500;

e=100;
t=1;

while e>epsilon && t<time
    temp=psi(gama_t)-ones(k,1)*sum(psi(gama_t),1)+(X*log(beta'+realmin)./(Ns*ones(1,k)))';
    maxtemp=max(temp,[],1);
    phi_tt=exp(temp+ones(k,1)*abs(maxtemp));
    phi_tt=phi_tt./(ones(k,1)*sum(phi_tt,1));
    
    gama_tt=alpha*ones(1,M)+phi_tt.*(ones(k,1)*Ns');
   
    e1=sum(sum(abs(phi_tt-phi_t)))/sum(sum(phi_t));
    e2=sum(sum(abs(gama_tt-gama_t)))/sum(sum(gama_t));
    e=max(e1,e2);
    
    phi_t=phi_tt;
    gama_t=gama_tt;
%     disp(['t=',int2str(t),', e1,e2,e:',num2str(e1),',',num2str(e2),',',num2str(e)]);
    t=t+1;
end

