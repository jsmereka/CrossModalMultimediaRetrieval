function [Pw_z,Pz_d] = getEMstep(X,Pw_d,Pw_z,Pz_d,beta)

if(nargin < 4)
    beta = 1;
end

% [Pw_z,Pz_d] = mex_EMstep(X,Pw_d,Pw_z,Pz_d,beta);

%[Pw_z1,Pz_d1] = mex_EMstep(X,Pw_d,Pw_z,Pz_d,beta);


[Ntopics,Ndocs] = size(Pz_d);

sX = full(sum(sum(X)));

for i=1:Ntopics
    %   [XPz_dw,sumXPz_dw] = mex_EMstep_old(X,Pw_d,Pw_z(:,i),Pz_d(i,:),beta);
    %   equiv with
    XPz_dw = X .* sparse((Pw_z(:,i) * Pz_d(i,:))./Pw_d);
    if(beta ~= 1)
        XPz_dw = (XPz_dw).^beta;
    end
    Pw_z(:,i) = sum(XPz_dw,2) ./ sum(sum(XPz_dw));
    Pz_d(i,:) = sum(XPz_dw,1) ./ sX;
end
% foo = 4;