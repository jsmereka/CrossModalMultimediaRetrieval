function txts = filter_stopwords(txts, stoplistfile, remove_nums)

if(nargin < 3)
    remove_nums = 0;
end

% load in stopwords file
fid = fopen(stoplistfile,'r');
stopwords = textscan(fid,'%s'); stopwords = lower(stopwords{1});
fclose(fid);

stopwords{end+1} = '';
stopwords{end+1} = ' ';

for i = 1:length(txts)
    txts{i} = lower(txts{i});
    if(remove_nums)
        txts{i} = regexp(txts{i},'[^\d]+','match');
        txts{i} = txts{i}(not(cell2mat(cellfun(@isempty,txts{i},'UniformOutput',false))));
        txts{i} = cellfun(@(x) char(cell2mat(x)),txts{i},'UniformOutput',false);
    end
    % remove characters that are not a->z
    txts{i} = regexp(txts{i},'[a-z]+','match');
    txts{i} = txts{i}(not(cell2mat(cellfun(@isempty,txts{i},'UniformOutput',false))));
    txts{i} = cellfun(@(x) char(cell2mat(x)),txts{i},'UniformOutput',false);
    % remove stop words
    txts{i} = txts{i}(not(ismember(txts{i},stopwords)));
end
