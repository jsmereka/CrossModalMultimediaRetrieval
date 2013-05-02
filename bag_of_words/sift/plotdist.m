function plotdist(im,neighbors)


imSize = [size(im,1) size(im,2)] ;

% plot
[u,v] = meshgrid(1:imSize(2),1:imSize(1)) ;
[v_,u_] = ind2sub(imSize, neighbors) ;

% avoid cluttering the plot too much
u = u(1:3:end,1:3:end) ;
v = v(1:3:end,1:3:end) ;
u_ = u_(1:3:end,1:3:end) ;
v_ = v_(1:3:end,1:3:end) ;

figure; clf ; imagesc(im) ; axis off image ;
hold on ; h = quiver(u,v,u_-u,v_-v,0) ; colormap gray ;