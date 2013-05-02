function [WordOccurance, ClassLabel] = format_txt_data(txts,words,cat_ids,freqcutoff,matvec)


if(nargin <= 3)
    freqcutoff = 0;
end

if(nargin <= 4)
    matvec = 1; % if(1) return a matrix, otherwise vector
end

if(matvec)
    WordOccurance = sparse(length(txts),length(words));
else
    totalwords = sum(cell2mat(cellfun(@(x) length(x),txts,'UniformOutput',false)));
    WordOccurance = zeros(1,totalwords);
    startidx = 0; endidx = 0;
end
if(nargout > 1)
    if(matvec)
        ClassLabel = zeros(length(txts),max(cat_ids)-1);
    else
        ClassLabel = WordOccurance;
    end
end

for i = 1:length(txts)
    tmp = cell2mat(cellfun(@(x) find(strcmp(words,x)), txts{i}, 'UniformOutput', false));
    if(matvec)
        if(freqcutoff == 0)
            WordOccurance(i,:) = hist(tmp,1:length(words));
        else
            tmp2 = hist(tmp,1:length(words));
            tmp2(tmp2 > freqcutoff) = 0;
            WordOccurance(i,:) = tmp2;
            clear tmp2;
        end
        if(nargout > 1)
            if(cat_ids(i) ~= max(cat_ids))
                ClassLabel(i,cat_ids(i)) = 1;
            end
        end
    else
        startidx = endidx + 1; endidx = startidx + length(tmp) - 1;
        WordOccurance(startidx:endidx) = tmp;
        if(nargout > 1)
            ClassLabel(startidx:endidx) = i;
        end
    end    
end
clear tmp;

    