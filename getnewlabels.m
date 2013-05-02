function labelindex = getnewlabels(prevlabel,predicts,topics)



newlabels = zeros(topics,topics);
for k = 1:topics
    newlabels(k,:) = hist(predicts(find(prevlabel == k)),1:topics);
end
labelindex = zeros(topics,1);
vec = 1:topics;
for k = 1:topics
    for q = 1:topics
        [r,c] = find(newlabels == max(max(newlabels)),1);
        if(labelindex(r) == 0)
            if(vec(c) ~= 0)
                labelindex(r) = c;
                vec(c) = 0;
            end
        end
        newlabels(r,c) = 0;
    end
end

if(sum(labelindex == 0) > 0)
    vec = 1:topics;
    labelindex(not(ismember(labelindex,vec))) = vec(not(ismember(vec,labelindex)));
end