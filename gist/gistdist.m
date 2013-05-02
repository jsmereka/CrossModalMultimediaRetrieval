

% L2
    dist = sum((gist - repmat(gistQuery, [size(gist,1) 1])).^2,2);
    [dist,j] = sort(dist);