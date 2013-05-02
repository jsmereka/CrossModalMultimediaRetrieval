function [distparam, angleparam, trndist, trnangles, tdist, tangle, xdist, xangle] = train_SIFT(imgs,usrs,indexes,featinfo)

trnangles = []; trndist = [];

for i = 1:length(usrs)
    for j = 1:length(usrs{i})
        if(ismember(usrs{i}(j),indexes))
            im1 = imgs{find(usrs{i}(j) == indexes)};
            for k = j+1:length(usrs{i})
                if(ismember(usrs{i}(k),indexes))
                    im2 = imgs{find(usrs{i}(k) == indexes)};
                    matches = vl_ubcmatch(im1{2},im2{2},featinfo.matchthresh);

                    xa = im1{1}(1,matches(1,:));
                    xb = im2{1}(1,matches(2,:)) + featinfo.imageSize(2);
                    ya = im1{1}(2,matches(1,:));
                    yb = im2{1}(2,matches(2,:));
                    theangles = abs(atan2(yb - ya, xb - xa) * 180 / pi);
                    distbetween = sqrt((xa-(xb-featinfo.imageSize(2))).^2 + (ya - yb).^2);
                    
                    trnangles = [trnangles, theangles];
                    trndist = [trndist, distbetween];
                end
            end
        end
    end
end


tmp = lognfit(trnangles(trnangles>0));
angleparam.mu = tmp(1); angleparam.sig = tmp(2);
angleparam.mode = exp(angleparam.mu - angleparam.sig.^2);
tmp = lognfit(trndist(trndist>0));
distparam.mu = tmp(1); distparam.sig = tmp(2);
distparam.mode = exp(distparam.mu - distparam.sig.^2);

xdist = 0:max(trndist)/(numel(trndist)-1):max(trndist);
tdist = lognpdf(xdist,distparam.mu,distparam.sig);
tdist(xdist<=distparam.mode) = tdist(max(find(xdist<=distparam.mode)));

xangle = 0:max(trnangles)/(numel(trnangles)-1):max(trnangles);
tangle = lognpdf(xangle,angleparam.mu,angleparam.sig);
tangle(tangle<=angleparam.mode) = angleparam.mode;
tangle(xangle<=angleparam.mode) = tangle(max(find(xangle<=angleparam.mode)));