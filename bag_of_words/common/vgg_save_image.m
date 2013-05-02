% vgg_save_image  Saves image, incl. formats mit, ppm, pgm.
%
% vgg_save_image(X, filename, format [,opt])
% The function behaves as imwrite(filename,X,format[,opt]) but
% it can save formats 'mit', 'pgm', 'ppm'.
%
% See also vgg_load_image.

% T.Werner, Feb 2002

function vgg_save_image(X, filename, format, varargin)

switch format
   case 'mit'
      vgg_save_image_mit(filename,X);
   case 'ppm'
      obs
      vgg_save_image_ppm(filename,X);
   case 'pgm'
      obs
      vgg_save_image_pgm(filename,X);
   otherwise
      obs
      imwrite(X,filename,format,varargin{:});
end

return

%%%%%%%%%%%%%%%%%

function obs
warning('This function is OBSOLETE and will soon be removed - imwrite in Matlab 6.5 supports this format.');
return