% VL_DEMO_SIFT_MATCH  Demo: SIFT: basic matching

clear; close all; clc;

Root = 'C:\Users\Jon\Documents\MATLAB\Ocular\PDM\';

warning off all;
rmpath(fullfile(Root,'features','sift','glnx86'));
rmpath(fullfile(Root,'features','sift','glnxa64'));
rmpath(fullfile(Root,'features','sift','win32'));
rmpath(fullfile(Root,'features','sift','win64'));
addpath(fullfile(Root,'features','sift','src'));

if(isunix)
    showfigs = 0;
    switch computer
        case 'GLNX86'
            addpath(fullfile(Root,'features','sift','glnx86'));
        case 'GLNXA64'
            addpath(fullfile(Root,'features','sift','glnxa64'));
        otherwise
            error('computer type not recognized');
    end
else
    showfigs = 0;
    switch computer
        case 'PCWIN'
            addpath(fullfile(Root,'features','sift','win32'));
        case 'PCWIN64'
            addpath(fullfile(Root,'features','sift','win64'));
        otherwise
            error('computer type not recognized');
    end
end

warning on all;
regsift = 0; threshcase = 0; siftflow = 1;

% Ia = imread('a.jpg');
% Ib = imread('b.jpg');
load(fullfile(Root,'info','tst_usrs.mat'));
binSize = 1; magnif = 1; boxthresh = 0.25; anglethresh = 20;
finscore = cell(1,3); trnangles = []; trndist = []; K = 10; nleaves = 1000;
SIFTparam.cellsize = 1;
SIFTparam.gridsize = 1;
SIFTparam.alpha = 2*255;
Param.SIFTparam.d = 40*255;
SIFTparam.gamma = 0.005*255;
SIFTparam.nlevels = 2;
SIFTparam.wsize = 2;
SIFTparam.topwsize = 10;
SIFTparam.nTopIterations = 4;
SIFTparam.nIterations = 5;
if(threshcase || ~regsift)
    trnlen = 1;
else
    trnlen = 2;
end
for q = 1:trnlen
    if(~threshcase && regsift)
        if(q == 1)
            % training
            disp('Training');
            nlen = 2; tst = 0;
        else
            disp('Testing');
            nlen = 3; tst = 1;
            if(regsift)
                tmp = lognfit(trnangles);
                angleMLE.mu = tmp(1); angleMLE.sig = tmp(2);
                angleMLE.mode = exp(angleMLE.mu - angleMLE.sig.^2);
                tmp = lognfit(trndist);
                distMLE.mu = tmp(1); distMLE.sig = tmp(2);
                distMLE.mode = exp(distMLE.mu - distMLE.sig.^2);
            else
                % dense sift

            end
        end
    else
        tst = 1; nlen = 3;
    end
    for n = 1:nlen
        switch(n)
            case 1
                qusr = usr1;
                tusr = usr1;
            case 2
                qusr = usr2;
                tusr = usr2;
            case 3
                qusr = usr1;
                tusr = usr2;
        end
        if(tst)
            finscore{n} = zeros(5,5);
        end
        for i = 1:5
            Ia = round(qusr{i}.*255); 
            imageSize = [size(Ia,1), size(Ia,2)];
            if(size(Ia,3) > 1)
                I1 = rgb2gray(Ia);
            else
                I1 = Ia;
            end
            if(regsift)
                [fa,da] = vl_sift(im2single(I1));
            else
                faa = mexDenseSIFT(im2single(I1),binSize,magnif);
                if(~siftflow)
                    [FaC,FaA] = vl_ikmeans(faa,K,'method','elkan');
%                     [treeA,FaA] = vl_hikmeans(faa,K,nleaves,'method','elkan');
                end
            end
            for j = 1:5

                Ib = round(tusr{j}.*255);
                if(size(Ib,3) > 1)
                    I2 = rgb2gray(Ib);
                else
                    I2 = Ib;
                end
                
                tic

                if(regsift)
                    [fb,db] = vl_sift(im2single(I2)) ;
                    [matches, scores] = vl_ubcmatch(da,db,1.15) ;
                    [drop, perm] = sort(scores, 'descend') ;
                    matches = matches(:, perm) ;
                    scores  = scores(perm) * 0.00001 ;

                    xa = fa(1,matches(1,:));
                    xb = fb(1,matches(2,:)) + imageSize(2);
                    ya = fa(2,matches(1,:));
                    yb = fb(2,matches(2,:));
                    theangles = abs(atan2(yb - ya, xb - xa) * 180 / pi);
                    distbetween = sqrt((xa-(xb-imageSize(2))).^2 + (ya - yb).^2);

                    if(~tst && i~=j)
                        trnangles = [trnangles, theangles];
                        trndist = [trndist, distbetween];
                    end
                    if(tst)
                        if(threshcase)
                            angle = theangles < anglethresh;
                            getboxtopx =  xa-((boxthresh*imageSize(2))/2);
                            getboxtopx(getboxtopx < 1) = 1;
                            getboxtopy =  ya-((boxthresh*imageSize(1))/2);
                            getboxtopy(getboxtopy < 1) = 1;
                            getboxbottomx = xa+((boxthresh*imageSize(2))/2);
                            getboxbottomx(getboxbottomx > imageSize(2)) = imageSize(2);
                            getboxbottomy = ya+((boxthresh*imageSize(1))/2);
                            getboxbottomy(getboxbottomy > imageSize(1)) = imageSize(1);
                            inthebox = (xb-imageSize(2))>getboxtopx & yb>getboxtopy & (xb-imageSize(2))<getboxbottomx & yb<getboxbottomy;
                            scores = scores(angle&inthebox);
                            s = 1 - (sum(scores)/length(scores));
                        else
                            angle = ones(size(theangles));
                            inthebox = ones(size(distbetween));
                            avec = theangles>0; dvec = distbetween>0;
                            if(sum(avec) > 0)
                                theangles((theangles <= angleMLE.mode)&(theangles > 0)) = angleMLE.mode;
                                tangle = lognpdf((theangles(avec)),angleMLE.mu,angleMLE.sig);
                                angle(avec) = tangle;
                            end
                            if(sum(dvec) > 0)
                                distbetween((distbetween <= distMLE.mode)&(distbetween > 0)) = distMLE.mode;
                                tinthebox = lognpdf((distbetween(dvec)),distMLE.mu,distMLE.sig);
                                inthebox(dvec) = tinthebox;
                            end
                            prob = angle.*inthebox;

                            s = sum(prob).*length(scores);
                            %s = 1 - (sum(scores.*(1-prob))/length(scores));
                        end
                        
                        finscore{n}(i,j) = s;
                        if(finscore{n}(i,j) == 0 || isnan(finscore{n}(i,j)) || isinf(finscore{n}(i,j)))
                            finscore{n}(i,j) = 0.000001;
                        end
                    end
                else
                    if(siftflow)
                        fbb = mexDenseSIFT(im2single(I2),binSize,magnif);
                        
                        [u,v,energylist]=SIFTflowc2f(faa,fbb,SIFTparam);
                        s = 1-energylist(1).data(end)/100000; % smallest energy
                    else

                        fbb = mexDenseSIFT(im2single(I2),binSize,magnif);
                        
                        % project fbb onto faa clusters
                        FbA = vl_ikmeanspush(fbb,FaC);
%                         FbA = vl_hikmeanspush(treeA,fbb);
                        
                        im = single(abs(FbA - FaA));
                        
                        if(all(not(im)))
                            s = 1;
                        else
                            im2 = reshape(single(im(1,:)>0),imageSize(1),imageSize(2));
                            [distanceTransform, neighbors] = vl_imdisttf(im2);
                            s = 1 - sum(distanceTransform(:));
%                             s = 1 - sum(im(:));
                        end
                    end
                    
                    finscore{n}(i,j) = s;
                    if(finscore{n}(i,j) == 0 || isnan(finscore{n}(i,j)) || isinf(finscore{n}(i,j)))
                        finscore{n}(i,j) = 0.000001;
                    end
                end
                
                toc
            end
        end
    end
end

[rate, thresh] = get_eer([finscore{1}(:);finscore{2}(:)],finscore{3}(:));
fprintf(1,'EER: %.2f%%\n', rate);

if(showfigs)
    plotsift(Ia,Ib,fa,fb,matches)
end
