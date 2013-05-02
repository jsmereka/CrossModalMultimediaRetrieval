% show top words in each topic

function [] = topicshow(phi, voc, N)

[K, W] = size(phi);

for k = 1:K
    [a, b] = sort(phi(k,:), 'descend');
    for i = 1:N
        fprintf(1, '%s ', voc{b(i)});
    end; 
    fprintf(1, '\n');
end