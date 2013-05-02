% [Pw_z,Pz_d] = pLSA_EMstep(X,Pw_z,Pz_d,beta,Pw_d);
%
% computes one step of EM and updates P(w|z) and P(z|d) where
%
% X		Term x document matrix
% Pw_z		conditional word probabilities
% Pz_d 		topic activation of documents
% beta		(optional) temperature needed for TEM
% Pw_d		(optional) needed to compute the normalization 
%
% Peter Gehler (pgehler@tuebingen.mpg.de) 
function [Pw_z,Pz_d] = pLSA_EMstep(X,Pw_z,Pz_d,beta,Pw_d);


if nargin < 4 
  beta = 1;
end

if nargin < 5
  Pw_d = getPw_d(X,Pw_z,Pz_d,beta);

  %equiv with:
  %Pw_d = zeros(Nwords,Ndocs);
  %for i=1:Ntopics
  % Pw_d = Pw_d + Pw_z(:,i) * Pz_d(i,:);
  %end
end

% [Pw_z,Pz_d] = mex_EMstep(X,Pw_d,Pw_z,Pz_d,beta);
[Pw_z,Pz_d] = getEMstep(X,Pw_d,Pw_z,Pz_d,beta);


return;
