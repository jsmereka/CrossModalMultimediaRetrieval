function [sections, centers, center_shift] = extract_secs_large(im, rows, columns, ratio, overlap, view)
if(nargin == 4)
    view = 0; overlap = 0;
elseif(nargin == 5)
    view = 0;
end

% Extracts the local regions from a whole image (or texture representation) 
%
% Inputs
%   im     : either a segmented image (M by N) or a texture representation (A by B by K)
%   ratios : specifies the partition into local regions
% Output
%   sections : cell array containing the local regions of the input 'im';
%              if 'im' is 2D, each local region is a 2D array; 
%              if 'im' is 3D, each local region is a 3D array

% view = 1, show image, view = 2 show empty image with centers

if(ratio == 1 && overlap == 0) % just use extract sections code
    [sections, centers] = extract_secs(im, rows, columns, view);
    center_shift = {};
else
    if(overlap > 0)
        [t1, t2, center_shift_1] = extract_secs_larger(im, rows, columns, overlap+1, view);
        clear t1 t2;
        [sections, centers, center_shift_2] = extract_secs_larger(im, rows, columns, ratio+overlap, view);
        center_shift = cell(1,length(center_shift_2));
        for i = 1:length(center_shift_1)
            center_shift{i} = center_shift_2{i} - center_shift_1{i};
        end
    else
        [sections, centers, center_shift] = extract_secs_larger(im, rows, columns, ratio, view);
    end
        
end


end

function [sections, centers] = extract_secs(im, rows, columns, view)
    if(nargin == 3)
        view = 0;
    end

    [m, n, q] = size(im);
    horz = floor(n / columns);
    vert = floor(m / rows);
    offset = [m - vert*rows, n - horz*columns];
    if(view == 1)
        figure; imshow(im);
    elseif(view == 2)
        figure; imshow(ones(size(im)));
    end
    ct = 1;
    sections = cell(1,rows*columns);
    centers = sections;
    for j = 1 : rows
        if(offset(1) < rows)
            s(1) = ((j-1)*vert)+1 + floor(offset(1)/2); 
            s(2) = j*vert + round(offset(1)/2);
        else
            s(1) = ((j-1)*vert)+1 + floor(offset(1)/rows); 
            s(2) = j*vert + round(offset(1)/rows);
        end
        if(s(2) > size(im,1))
            s(2) = size(im,1);
        end
        for k = 1 : columns
            if(offset(2) < columns)
                f(1) = ((k-1)*horz)+1 + floor(offset(2)/2); 
                f(2) = k*horz + floor(offset(2)/2);
            else
                f(1) = ((k-1)*horz)+1 + floor(offset(2)/columns); 
                f(2) = k*horz + floor(offset(2)/columns);
            end
            if(f(2) > size(im,2))
                f(2) = size(im,2);
            end
            sections{ct} = im(s(1):s(2),f(1):f(2),:);
            centers{ct} = [s(1) + floor((s(2)-s(1))/2), f(1) + floor((f(2) - f(1))/2)];
            if(view > 0)
                rectangle('Position',[f(1), s(1), horz, vert],'edgecolor','r','linewidth',2);
                if(view == 2)
                    rectangle('Position',[centers{ct}(2), centers{ct}(1), 3, 3],'edgecolor','r','linewidth',1);
                end
            end
            ct = ct + 1;
        end
    end

end


function [sections, centers, center_shift] = extract_secs_larger(im, rows, columns, ratio, view)
    if(nargin == 4)
        view = 0;
    end

    [m, n, q] = size(im);
    horz = floor(n / columns);
    vert = floor(m / rows);
    v = floor(vert*ratio)-vert;
    h = floor(horz*ratio)-horz;
    s = zeros(1,2); f = zeros(1,2);
    c = floor([horz/2, vert/2]);
    offset = [m - vert*rows, n - horz*columns];
    if(view > 0)
        figure; imshow(ones(round(ratio.*size(im,1)),round(ratio.*size(im,2))));
    end
    ct = 1; centoffset = [0,0];
    centers = cell(1,rows*columns); center_shift = centers; sections = centers;
    for j = 1 : rows
        if(offset(1) < rows)
            addtoR = floor(offset(1)/2);
        else
            addtoR = floor(offset(1)/rows);
        end
        if(j ~= rows)
            if(j ~= 1)
                s(1) = ((j-1)*vert)-floor(v/2) +1 + addtoR;
                s(2) = (j*vert)+floor(v/2) +1 + addtoR;
                if(j == 2)
                    centoffset(1) = floor(v/2) + v;
                else
                    centoffset(1) = v + centoffset(1);
                end
            else
                s(1) = 1 + addtoR;
                s(2) = vert+v + addtoR;
                centoffset(1) = 0;
            end
        else
            s(1) = ((j-1)*vert)-v +1 + addtoR;
            s(2) = j*vert +1 + addtoR;
            centoffset(1) = floor(1.5*v) + centoffset(1);
        end
        if(s(2) > size(im,1))
            s(2) = size(im,1);
        end
        if(s(1) < 1)
            s(1) = 1;
        end
        for k = 1 : columns
            if(offset(2) < columns)
                addtoC = floor(offset(2)/2);
            else
                addtoC = floor(offset(2)/columns);
            end
            if(k ~= columns)
                if(k ~= 1)
                    f(1) = ((k-1)*horz) - floor(h/2) +1 + addtoC;
                    f(2) = (k*horz) + floor(h/2) +1 + addtoC;
                    if(k == 2)
                        centoffset(2) = floor(h/2) + h;
                    else
                        centoffset(2) = h + centoffset(2);
                    end
                else
                    f(1) = 1 + addtoC;
                    f(2) = horz+h + addtoC;
                    centoffset(2) = 0;
                end
            else
                f(1) = ((k-1)*horz)-h +1 + addtoC;
                f(2) = k*horz +1 + addtoC;
                centoffset(2) = floor(1.5*h) + centoffset(2);
            end
            if(f(2) > size(im,2))
                f(2) = size(im,2);
            end
            if(f(1) < 1)
                f(1) = 1;
            end
            sections{ct} = im(s(1):s(2),f(1):f(2),:);
            centers{ct} = [(c(2)+(j-1)*vert)+addtoR+centoffset(1), (c(1)+(k-1)*horz)+addtoC+centoffset(2)];
            center_shift{ct} = [s(1) + round((s(2)-s(1))/2)+centoffset(1), f(1) + round((f(2) - f(1))/2)+centoffset(2)] - centers{ct};
            ct = ct + 1;
            if(view > 0)
                if(view == 2)
                    rectangle('Position',[centers{ct-1}(2), centers{ct-1}(1), 1, 1],'edgecolor','g');
                    rectangle('Position',[centers{ct-1}(2)+center_shift{ct-1}(2), centers{ct-1}(1)+center_shift{ct-1}(1), 1, 1],'edgecolor','b');
                end
                rectangle('Position',[((k-1)*horz)+1+addtoC+centoffset(2), ((j-1)*vert)+1+addtoR+centoffset(1), horz, vert],'edgecolor','r');
                rectangle('Position',[f(1)+centoffset(2), s(1)+centoffset(1), f(2)-f(1), s(2)-s(1)],'edgecolor','k');
            end
        end
    end
end
