function plotkmeans(img,Assign1,Assign2,K)


im1 = zeros(size(img,1)*size(img,2),1); im2 = im1; im11 = im1; im22 = im1;
prevstart = 1; prevstart2 = 1;
for k=1:K
  sel  = find(Assign1 == k);
  selt = find(Assign2 == k);
  im1(prevstart:prevstart+numel(sel)-1) = img(sel); 
  im11(prevstart:prevstart+numel(sel)-1) = k.*ones(numel(sel),1);
  im2(prevstart2:prevstart2+numel(selt)-1) = img(selt); 
  im22(prevstart2:prevstart2+numel(selt)-1) = k.*ones(numel(selt),1);
  prevstart = prevstart+numel(sel); prevstart2 = prevstart2+numel(selt);
end

im1 = reshape(im1./255,size(img,1),size(img,2)); im11 = reshape(im11,size(img,1),size(img,2));
im2 = reshape(im2./255,size(img,1),size(img,2)); im22 = reshape(im22,size(img,1),size(img,2));

figure; subplot(1,2,1); imshow(abs(im1-im2)); subplot(1,2,2); imagesc(abs(im11-im22)); 