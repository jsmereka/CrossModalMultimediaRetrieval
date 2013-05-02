function [alpha,beta,eta]=fastDldaMstep(alpha_last,eta_last,phi,gama,X,Y,lap)
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
%   alpha_last:     k*1; alpha from last iteration
%   eta_last:       k*(c-1), eta from last iteration, regression parameter for c-1 classes, if
%                   #class=#cluster, c=k;
%   phi:            k*M;
%   gama:           k*M;
%   X:              M*V; M docs, each is represented as word counts
%   Y:              M*(c-1); labels for M docs, each row is a unit vector
%   lap:            parameter for smoothing.
%
% Ouptput:
%   alpha:      k*1;
%   beta:       k*V;
%   eta:        k*(c-1)
%------------------------------------



[k,M]=size(phi);
[M,V]=size(X);
c=size(Y,2)+1; % number of classes
Ns=sum(X,2);    %Ms: the number of words in each document

%xis: 1*M, xi for each doc
xis=1+sum(phi.*(sum(exp(eta_last),2)*ones(1,M)),1);

% beta
for i=1:k
    tempphi=phi(i,:)'*ones(1,V);
    beta(i,:)=sum(tempphi.*X,1);
end
beta=beta+lap;
beta=beta./(sum(beta,2)*ones(1,V));

% eta
s1=phi*Y;
s2=sum(phi./(ones(k,1)*xis),2);
eta=log(s1./(s2*ones(1,c-1)+realmin)+realmin);


% alpha
alpha_t=alpha_last;
epsilon=0.001;
time=500;

t=0;
e=100;
psiGama=psi(gama);
psiSumGama=psi(sum(gama,1));
while e>epsilon&&t<time
    g=sum((psiGama-ones(k,1)*psiSumGama),2)+M*(psi(sum(alpha_t))-psi(alpha_t));
    h=-M*psi(1,alpha_t);
    z=M*psi(1,sum(alpha_t));
    c=sum(g./h)/(1/z+sum(1./h));
    delta=(g-c)./h;

    tao=1;
    alpha_tt=alpha_t-delta;
    while (length(find(alpha_tt<=0))>0)
        tao=tao/2;
        alpha_tt=alpha_t-tao*delta;
    end
    e=sum(abs(alpha_tt-alpha_t))/sum(alpha_t);
    
    alpha_t=alpha_tt;

    t=t+1;
end
alpha=alpha_t;