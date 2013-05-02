function [alpha_t,beta_t,eta_t,phi_t,gama_t,logProb_time,perplexity_time]=learnFastDlda(X,Y,alpha,beta,eta,lap,flag,verbose)
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
%   X:          M*V; M docs, each doc is represented as word occurrence
%   Y:          M*(c-1); M*(c-1), each row is the class label for one doc.  
%               The ith dimension with value 1 indicates the doc class is
%               i. If all dimensions are 0, the doc class is c.
%   alpha:      k*1; Dirichlet Distribution
%   beta:       k*V; Discrete Distributions for k topics
%   eta:        k*(c-1), regression parameter for c-1 classes, if
%               #class=#cluster, c=k;
%   lap:        laplacian smoothing parameter
%   flag:       scaler, if flag=1, use the change of perplexity to check the convergence,
%               if flag=0, use the change of parameter to check the convergence
%
% Ouptput:
%   alpha_t:                        k*1
%   beta_t:                         k*V
%   eta_t:                          k*(c-1)
%   phi_t:                          k*M
%   gama_t:                         k*M
%   logProb_time, perplexity_time:  log-likelihood and perplexity over
%                                   iterations
%---------------------------------------------------

if(nargin < 8)
    verbose = 1;
end

[M,V] = size(X);
[k,V]=size(beta);

alpha_t=alpha;
beta_t=beta;
eta_t=eta;
phi_t=ones(k,M)/k;

clear alpha beta eta;

perplexity_t=1;

logProb_time=[];
perplexity_time=[];

epsilon=0.001;
time=500;

e=100;
t=1;
if(verbose)
    disp(['learning Fast DLDA'])
end
while e>epsilon && t<time
    % E-step
    [phi_tt,gama_tt]=fastDldaEstep(alpha_t,beta_t,eta_t,phi_t,X,Y);
    
    % compute perplexity if flag=1
    if flag==1
        [logProb_tt,perplexity_tt]=fastDldaGetPerp(X,Y,alpha_t,beta_t,eta_t,phi_tt,gama_tt);
        logProb_time=[logProb_time,logProb_tt];
        perplexity_time=[perplexity_time,perplexity_tt];
    end
    
    % M-step
    [alpha_tt,beta_tt,eta_tt]=fastDldaMstep(alpha_t,eta_t,phi_tt,gama_tt,X,Y,lap);
    
    % difference from the previouse iteration
    if flag==1
        if perplexity_tt==Inf||perplexity_t==Inf
            e=100;
        else
            e=abs(perplexity_tt-perplexity_t)/perplexity_t;
        end
        if(verbose)
            disp(['t=',int2str(t),' error= ',num2str(e), ' perplexity=',num2str(perplexity_tt)]);
        end
        logProb_t=logProb_tt;
        perplexity_t=perplexity_tt;
    else
        e1=sum(sum(sum(abs(beta_t-beta_tt))))/sum(sum(sum(beta_t)));
        e2=0;%norm(alpha_t-alpha_tt);
        e=max([e1,e2]);
        if(verbose)
            disp(['t=',int2str(t),' error= ',num2str(e)]);
        end
    end
    
   
    alpha_t=alpha_tt;
    beta_t=beta_tt;
    eta_t=eta_tt;
    phi_t=phi_tt;
    gama_t=gama_tt;
    
    t=t+1;
    
end


