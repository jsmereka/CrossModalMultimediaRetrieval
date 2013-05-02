function [phi, theta] = LaLDAssiBPtrain(WD, AD, T, ALPHA, BETA, OUTPUT)

% WD is a WxD sparse matrix (data)
% AD is a AxD sparse matrix (label)
% T is the number of iterations
% phi is a AxW matrix
% theta is a AxD matrix
% ALPHA and BETA are hyperparameters
% OUTPUT == 1 showi the number of iterations

if (~issparse(WD) || ~issparse(AD) || (T <= 0) || (ALPHA <= 0) || (BETA <= 0)) 
    error('Input error'); 
end

% get word and doc index
[W, D] = size(WD);
[A, D] = size(AD);
if A > 50
    error('The number of labels is too large');
end
[wi, di, xi] = find(WD);

if (max(wi) < W)
    wi(1) = W;
end


% random initialization
theta = rand(A, D);
theta = normalise(theta .* AD, 1);
phi = normalise(rand(A, W), 2);

% simplified belief propagation
for t = 1:T
    if OUTPUT >= 1
        if (mod(t,10) == 0)
            fprintf(1, '\tIteration %d of %d: %f\n', t, T, full(perp));
        end
    end 
    mu = normalise(theta(:, di) .* phi(:, wi), 1);
    for k = 1:A
        md(k, :) = accumarray(di, xi' .* mu(k, :));
        mw(k, :) = accumarray(wi, xi' .* mu(k, :));
    end
    theta = normalise((md + ALPHA) .* AD, 1);
    phi = normalise(mw + BETA, 2);
    perp = exp(sum(-log(sum(theta(:, di) .* phi(:, wi), 1)) .* xi')/sum(xi));
end

function [M, z] = normalise(A, dim)
% NORMALISE Make the entries of a (multidimensional) array sum to 1
% [M, c] = normalise(A)
% c is the normalizing constant
%
% [M, c] = normalise(A, dim)
% If dim is spexified, we normalise the spexified dimension only,
% otherwise we normalise the whole array.

if nargin < 2
    z = sum(A(:));
    % Set any zeros to one before dividing
    % This is valid, since c=0 => all i. A(i)=0 => the answer should be 0/1=0
    s = z + (z == 0);
    M = A / s;
elseif dim == 1 % normalize each column
    z = sum(A);
    s = z + (z==0);
    %M = A ./ (d'*ones(1,size(A,1)))';
    M = A ./ repmat(s, size(A,1), 1);
else
    % Keith Battocchi - v. slow because of repmat
    z = sum(A,dim);
    s = z + (z == 0);
    L = size(A,dim);
    d = length(size(A));
    v = ones(d,1);
    v(dim) = L;
    %c=repmat(s,v);
    c = repmat(s,v');
    M = A./c;
end