function showMSIFTdemo

addtopath; close all;

im1=imread('true1.jpg');
im2=imread('true2.jpg');
im3=imread('false2.jpg');

showdemo(im1,im2,[384,384],2,20,0.35);
showdemo(im1,im3,[384,384],2,20,0.35);



function showdemo(a,b,imgsize,matchthresh,anglethresh,boxthresh)   
    
if(size(a,1) ~= imgsize(1) && size(a,2) ~= imgsize(2) && (imgsize(1) > 0 && imgsize(2) > 0))
    a = imresize(a, imgsize(1:2));
end
if(size(a,3) > 1)
    a = rgb2gray(a);
end
if(size(b,1) ~= imgsize(1) && size(b,2) ~= imgsize(2) && (imgsize(1) > 0 && imgsize(2) > 0))
    b = imresize(b, imgsize(1:2));
end
if(size(b,3) > 1)
    b = rgb2gray(b);
end

[keypointsa,descriptorsa] = vl_sift(im2single(a));
[keypointsb,descriptorsb] = vl_sift(im2single(b));
[matches,scores] = vl_ubcmatch(descriptorsa,descriptorsb,matchthresh);
plotsift(a,b,keypointsa,keypointsb,matches); title('Initial Match');

if(~isempty(matches))
    xa = keypointsa(1,matches(1,:));
    xb = keypointsb(1,matches(2,:)) + imgsize(2);
    ya = keypointsa(2,matches(1,:));
    yb = keypointsb(2,matches(2,:));
    theangles = abs(atan2(yb - ya, xb - xa) * 180 / pi);
    
    angles = theangles < anglethresh;
    
    getboxtopx =  xa-((boxthresh*imgsize(2))/2);
    getboxtopx(getboxtopx < 1) = 1;
    getboxtopy =  ya-((boxthresh*imgsize(1))/2);
    getboxtopy(getboxtopy < 1) = 1;
    getboxbottomx = xa+((boxthresh*imgsize(2))/2);
    getboxbottomx(getboxbottomx > imgsize(2)) = imgsize(2);
    getboxbottomy = ya+((boxthresh*imgsize(1))/2);
    getboxbottomy(getboxbottomy > imgsize(1)) = imgsize(1);
    
    inthebox = (xb-imgsize(2))>getboxtopx & yb>getboxtopy & (xb-imgsize(2))<getboxbottomx & yb<getboxbottomy;
%     oldmatchlen = size(matches,2);
    matches = matches(:,angles&inthebox);
%     scores = scores(angles&inthebox);
%     if(~isempty(scores))
%         score = -mode(scores)/length(scores);
%         %score = length(scores); %m-SIFT default
%         xa = keypointsa(1,matches(1,:));
%         xb = keypointsb(1,matches(2,:));
%         ya = keypointsa(2,matches(1,:));
%         yb = keypointsb(2,matches(2,:));
%         nodescores = [xa;ya;xb;yb;scores;oldmatchlen.*ones(size(scores))];
%     else
%         score = -500000; nodescores = [];
%     end
end

plotsift(a,b,keypointsa,keypointsb,matches); title('Match After Constraints');

end

end
