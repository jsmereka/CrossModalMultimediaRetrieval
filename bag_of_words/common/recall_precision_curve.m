function [curve,opP,Area,Threshold]=recall_precision_curve(X,Total_TP,Recall_Value)

AVERAGE_PRECISION_VALUES = [0:0.1:1];

if (nargin==2)
    Recall_Value=0;
end

if (Recall_Value==0)
    AVERAGE_PRECISION=1;
end 

if (size(X,1)>size(X,2))
   X=X';
end

[dum,ind]=sort(-X(1,:));

X=X(:,ind);

for a=1:length(X)

   y(a)=sum(X(2,1:a))/Total_TP;
   
   x(a)=length(find(X(2,1:a)==1))/a;
     
   t(a)=X(1,a);
   
end

if AVERAGE_PRECISION
   
   for q=1:length(AVERAGE_PRECISION_VALUES)
      good_ind = find(y>=AVERAGE_PRECISION_VALUES(q));
      if (~isempty(good_ind))
	 AP(q)=max(x(good_ind));
      else
	 AP(q)=0;
      end
   end
   
   opP = mean(AP);
   Threshold = 0;
else
   %%% Get particular point on RPC curve
   [dum,ind]=min(abs(y-Recall_Value));
   opP=x(ind);
   Threshold=t(ind);
end

Area=trapz(y,x);

curve = [y;x]';
