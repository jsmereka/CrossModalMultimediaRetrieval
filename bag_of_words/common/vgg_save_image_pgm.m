function vgg_SaveImagePGM(filename,X)
% function vgg_SaveImagePGM(filename,X)
%
% Save X to filename as PGM.
% Range of X is expected to be [0..255].

fd = fopen(filename, 'w');
if fd < 0, error(['Failed to open "' filename '"']);end
[h,w] = size(X);
fprintf(fd,'P5\n%i %i\n255\n',w,h);
ucx = X';
ucx = ucx(:); 
fwrite(fd, ucx, 'uchar');
fclose(fd);
