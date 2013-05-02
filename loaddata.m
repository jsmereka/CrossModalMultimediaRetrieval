function [cat_ids, imgs, txts] = loaddata(root,set,imgsize,getimgs,gettxts)

if(nargin == 2)
    imgsize = [];
elseif(nargin == 3)
    if(length(imgsize) ~= 2)
        imgsize = [];
    end
end

if(nargin < 4)
    getimgs = 0;
    gettxts = 0;
    if(nargout == 2)
        getimgs = 1;
        gettxts = 0;
    elseif(nargout == 3)
        getimgs = 1;
        gettxts = 1;
    end
elseif(nargin < 5)
    gettxts = 0;
    if(nargout == 2)
        gettxts = 0;
    elseif(nargout == 3)
        gettxts = 1;
    end
end

categories = 'categories.list';
% get set list
fileID = fopen(fullfile(root,set));
C = textscan(fileID, '%s %s %f');
fclose(fileID);
text_ids = C{1}; img_ids = C{2}; cat_ids = C{3}; clear C;
% get category mapping
fileID = fopen(fullfile(root,categories));
C = textscan(fileID, '%s');
fclose(fileID);
cat_name = C{1}; clear C;

if(gettxts)
    % load text
    txts = cell(1,length(cat_ids));
    ct = 1; idx = [];    
    for i = 1:length(cat_ids)
        try
            DOMnode = xmlread(fullfile(root,'texts',[text_ids{i} '.xml']));
            t = DOMnode.getElementsByTagName('text'); t = t.item(0);
            txts{i} = cell(t.getFirstChild.getData.split('[^\w]+'));
        catch e
            idx(ct) = i; ct = ct + 1;
        end
    end
    if(ct > 1) % error in reading some files
        ct = 1;
        fprintf(1,'Successfully loaded a total of %i files, %i unsuccessful - loading those now\n', length(cat_ids)-length(idx), length(idx));
        for i = 1:length(idx)
            try
                fid = fopen(fullfile(root,'texts',[text_ids{idx(i)} '.xml']),'r');
                C = textscan(fid,'%s','Delimiter','\n'); C = C{1};
                fclose(fid);
                startidx = 1+find(cell2mat(cellfun(@isempty,strfind(C,'<text>'),'UniformOutput', false))==0);
                endidx = find(cell2mat(cellfun(@isempty,strfind(C,'</text>'),'UniformOutput', false))==0)-1;
                tmp = C{startidx};
                for k = startidx+1:endidx
                    if(isempty(C{k}))
                        tmp = horzcat(tmp,' ');
                    else
                        tmp = horzcat(tmp,C{k});
                    end
                end
                C = tmp;
                tmp = regexp(C,'&(\w+);','match');
                if(isempty(tmp))
                    txts{idx(i)} = regexp(C,'[^\w]+','split')';
                else
                    C = strrep(C, '&hellip;', '');
                    C = strrep(C, '&mdash;', '-');
                    C = strrep(C, '&ndash;', '-');
                    C = strrep(C, '&minus;', '-');
                    C = strrep(C, '&prime;', '');
                    C = strrep(C, '&times;', '');
                    C = strrep(C, '&deg;', '');
%                     for k = 1:length(tmp)
%                         errmess{ct} = tmp{k}; ct = ct + 1;
%                     end
                    txts{idx(i)} = regexp(C,'[^\w]+','split')';
                end
            catch e
                fprintf('Unable to load %s\n',fullfile(root,'texts',[text_ids{idx(i)} '.xml']));
            end
        end
        ct = sum(cell2mat(cellfun(@isempty,txts,'UniformOutput',false)));
        if(ct == 0)
            fprintf(1,'Successfully loaded xml files\n');
        else
            fprintf(1,'%i xml files where unable to be loaded\n', ct);
        end
    else
        fprintf(1,'Successfully loaded xml files\n');
    end
else
    txts = {};
end

if(getimgs)
    % load images
    imgs = cell(1,length(cat_ids));
    ct = 1; idx = [];    
    for i = 1:length(cat_ids)
        try
            if(~isempty(imgsize))
                im = imread(fullfile(root,'images',cat_name{cat_ids(i)},[img_ids{i} '.jpg']));
                imgs{i} = imresize(im, imgsize);
            else
                imgs{i} = imread(fullfile(root,'images',cat_name{cat_ids(i)},[img_ids{i} '.jpg']));
            end
        catch e
            idx(ct) = i; ct = ct + 1;
        end
    end
    if(ct > 1) % error in reading some files
        fprintf(1,'Successfully loaded a total of %i files, %i unsuccessful - loading those now\n', length(cat_ids)-length(idx), length(idx));
    else
        fprintf(1,'Successfully loaded image files\n');
    end
else
    imgs = {};
end