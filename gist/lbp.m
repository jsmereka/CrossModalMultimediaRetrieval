function lbp_features = lbp(img,imgSize)
    if(size(img,3)==3)
        img = rgb2gray(img);
    end
    img = imresize(img,imgSize);
    img_neighborhoods = repmat(img,[1 1 8]);
    img_neighborhoods(2:end,2:end,1) = img(1:end-1,1:end-1);
    img_neighborhoods(2:end,:,2) = img(1:end-1,:);
    img_neighborhoods(2:end,1:end-1,3) = img(1:end-1,2:end);
    img_neighborhoods(:,2:end,4) = img(:,1:end-1);
    img_neighborhoods(:,1:end-1,5) = img(:,2:end);
    img_neighborhoods(1:end-1,2:end,6) = img(2:end,1:end-1);
    img_neighborhoods(1:end-1,:,7) = img(2:end,:);
    img_neighborhoods(1:end-1,1:end-1,8) = img(2:end,2:end);
    lbp_features = zeros(size(img));
    for i=1:8
        lbp_features = lbp_features+(2^i)*(img_neighborhoods(:,:,i)>img);
    end
end