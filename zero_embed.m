function nim1 = zero_embed(im1, a1, b1)

if(iscell(im1))
    nim1 = cell(size(im1));
    for i = 1:numel(im1)
        nim1{i} = do_work(im1{i},a1,b1);
    end
else
    nim1 = do_work(im1,a1,b1);
end
    
    
    
function nim = do_work(im,a,b)

[m, n, e] = size(im);

if(a == m && b == n)
    nim = im;
else
    if(a < m || b < n)
        if(a < m && b < n)
            nim1 = zeros(a,b,e);
        elseif(b < n)
            nim1 = zeros(m,b,e);
        elseif(a < m)
            nim1 = zeros(a,n,e);
        end
        mar1 = floor(abs(a - m) / 2);
        mar2 = floor(abs(b - n) / 2);
        if(e > 1)
            for i = 1:e
                if(a < m && b < n)
                    nim1(:,:,i) = im(mar1+1:a+mar1,mar2+1:b+mar2,i);
                elseif(b < n)
                    nim1(:,:,i) = im(:,mar2+1:b+mar2,i);
                elseif(a < m)
                    nim1(:,:,i) = im(mar1+1:a+mar1,:,i);
                end
            end
        else
            if(a < m && b < n)
                nim1 = im(mar1+1:a+mar1,mar2+1:b+mar2);
            elseif(b < n)
                nim1 = im(:,mar2+1:b+mar2);
            elseif(a < m)
                nim1 = im(mar1+1:a+mar1,:);
            end
        end
    else
        nim1 = im;
    end
    [m, n, e] = size(nim1);
    if(a > m || b > n)
        if(a > m && b > n)
            nim = zeros(a,b,e);
        elseif(b > n)
            nim = zeros(m,b,e);
        elseif(a > m)
            nim = zeros(a,n,e);
        end
        mar1 = floor(abs(a - m) / 2);
        mar2 = floor(abs(b - n) / 2);
        if(e > 1)
            for i = 1:e
                if(a > m && b > n)
                    nim(mar1 + 1 : mar1 + m, mar2 + 1 : mar2 + n, i) = nim1(:,:,i);
                elseif(b > n)
                    nim(:, mar2 + 1 : mar2 + n, i) = nim1(:,:,i);
                elseif(a > m)
                    nim(mar1 + 1 : mar1 + m, :, i) = nim1(:,:,i);
                end
            end
        else
            if(a > m && b > n)
                nim(mar1 + 1 : mar1 + m, mar2 + 1 : mar2 + n) = nim1;
            elseif(b > n)
                nim(:, mar2 + 1 : mar2 + n) = nim1;
            elseif(a > m)
                nim(mar1 + 1 : mar1 + m, :) = nim1;
            end
        end
    else
        nim = nim1;
    end
end