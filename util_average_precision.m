function ap = util_average_precision(recall, precision)

    mrecall = [0 ; recall ; 1];
    mprecision=[0 ; precision ; 0];
    
%     for i = numel(mprecision)-1:-1:1
%         mprecision(i) = max(mprecision(i), mprecision(i+1));
%     end
    
    i = find(mrecall(2:end) ~= mrecall(1:end-1))+1;
    ap = sum((mrecall(i)-mrecall(i-1)) .* mprecision(i));

%     indices = find(recall(2:end) > recall(1:end-1));
%     indices = indices + 1;
%     ap = mean(precision(indices));

end