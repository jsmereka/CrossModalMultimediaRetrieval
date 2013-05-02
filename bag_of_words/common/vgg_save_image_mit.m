function vgg_save_image_mit(filename,X)
% function vgg_save_image_mit(filename,X)
%
% Save X to filename as MIT (very obsolete!).
% Range of X is expected to be [0..255].

fd = fopen(filename, 'w');
if fd < 0, error(['Failed to open "' filename '"']);end
[h,w] = size(X);
shortheader = [1 8 w h];
ucharheader=[];
for s=shortheader
  hi = fix(s/256);
  lo = s - hi*256;
  ucharheader=[ucharheader lo hi];
end
fwrite(fd, ucharheader, 'uchar');
ucx = X';
ucx = ucx(:); 
fwrite(fd, ucx, 'uchar');
fclose(fd);
