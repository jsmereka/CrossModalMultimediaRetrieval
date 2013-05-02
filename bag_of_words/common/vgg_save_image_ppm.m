function vgg_save_image_ppm(filename, I);
%
% function vgg_save_image_ppm(filename, I);
%
% fsm@robots.ox.ac.uk
%

if ndims(I) ~= 3
  error('image must have three dimensions')
end
if size(I, 3) ~= 3
  error('last dimension must be 3')
end

fd = fopen(filename, 'w');
if fd < 0, error(['Failed to open "' filename '"']);end

fprintf(fd, 'P6\n');
% no #comments, please
fprintf(fd, '%d %d\n', size(I, 2), size(I, 1));
fprintf(fd, '255\n'); % assume maxval is 255
fwrite(fd, permute(I, [3 2 1]), 'uchar');

fclose(fd);

return;
