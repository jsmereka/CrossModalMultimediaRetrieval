function Pw_d = getPw_d(X,Pw_z,Pz_d,beta)

if(nargin == 3)
    beta = 1;
end


%Pw_d = mex_Pw_d(X,Pw_z,Pz_d);
%[tm1,tm2] = find(X);
%tmp = sub2ind(size(X),tm1,tm2);
Pw_d = sparse((Pw_z*Pz_d).^beta);
%Pw_d = sparse(tm1,tm2,tmp2(tmp),size(X,1),size(X,2));
% [Nwords,Ndocs] = size(X);
% Ntopics = size(Pw_z,2);
% Pw_d = sparse(Nwords,Ndocs);
% for i=1:Ntopics
%     Pw_d = Pw_d + (Pw_z(:,i) * Pz_d(i,:)).^beta;
% end
% %Pw_d = sparse(Pw_d);