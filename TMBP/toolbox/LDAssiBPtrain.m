function [phi, theta] = LDAssiBPtrain(WD, K, T, ALPHA, BETA, OUTPUT)

% WD is a WxD sparse matrix (data)
% K is the number of topixi
% T is the number of iterations
% phi is a KxW matrix
% theta is a KxD matrix
% ALPHA and BETA are hyperparameters
% OUTPUT == 1 showi the number of iterations

if (~issparse(WD) || (K <= 0) || (T <= 0) || (ALPHA <= 0) || (BETA <= 0))
    error('Input error');
end

% get word and doc index
[W, D] = size(WD);
[wi, di, xi] = find(WD);

if (max(wi) < W)
    wi(1) = W;
end

% index for accumarray function
% dx = [reshape(kron(ones(1,numel(di)),1:K),[],1) reshape(repmat(di',K,1),[],1)];
% wx = [reshape(kron(ones(1,numel(wi)),1:K),[],1) reshape(repmat(wi',K,1),[],1)];
% cx = reshape(repmat(xi',K,1),[],1);
% theta = accumarray(dx, cx .* mu(:));
% phi = accumarray(wx, cx .* mu(:));

% random initialization
mu = normalise(rand(K, nnz(WD)), 1);

% simplified belief propagation
for t = 1:T
    if OUTPUT >= 1
        if (mod(t,10) == 0)
            fprintf(1, '\tIteration %d of %d: %f\n', t, T, perp);
        end
    end
    for k = 1:K
        md(k, :) = accumarray(di, xi' .* mu(k, :));
        mw(k, :) = accumarray(wi, xi' .* mu(k, :));
    end
    theta = normalise(md + ALPHA, 1);
    phi = normalise(mw + BETA, 2);
    perp = exp(sum(-log(sum(theta(:, di) .* phi(:, wi), 1)) .* xi')/sum(xi));
    mu = normalise(theta(:, di) .* phi(:, wi), 1);
end

function [M, z] = normalise(A, dim)
% NORMALISE Make the entries of a (multidimensional) array sum to 1
% [M, c] = normalise(A)
% c is the normalizing constant
%
% [M, c] = normalise(A, dim)
% If dim is specified, we normalise the specified dimension only,
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