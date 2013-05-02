function [recall, precision] = util_calc_precision_recall_curve(ranking)

    cumulative = cumsum(ranking);
    nRelevant = sum(ranking);
    recall = zeros(size(ranking));
    precision = zeros(size(ranking));
    
    for i = 1:length(ranking)
        recall(i) = cumulative(i) / nRelevant;
        precision(i) = cumulative(i) / i;
    end

end