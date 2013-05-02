function [phi, theta] = LDAVBtrain(WD, K, T, E, ALPHA, BETA, OUTPUT)

% WD is a WxD sparse matrix (data)
% K is the number of topici
% T is the number of iterations
% E is the number of inner EM loops.
% phi is a KxW matrix
% theta is a KxD matrix
% ALPHA and BETA are hyperparameters
% OUTPUT == 1 showi the number of iterations

if (~issparse(WD) || (K <= 0) || (T <= 0) || (E <= 0) || (ALPHA <= 0) || (BETA <= 0)) 
    error('Input error'); 
end

[wi, di, ci] = find(WD);

% random initialization
mu = normalise(rand(K, nnz(WD)), 1);
for k = 1:K
    md(k, :) = accumarray(di, ci' .* mu(k, :));
    mw(k, :) = accumarray(wi, ci' .* mu(k, :));
end
phi = normalise(mw + BETA, 2);

% belief propagation
for t = 1:T
    if OUTPUT >= 1
        if (mod(t,10) == 0)
            fprintf(1, '\tIteration %d of %d: Perplexity %f\n', t, T, perp);
        end
    end     
    for e = 1:E
        mu = normalise(exp(psi(md(:, di) + ALPHA)) .* phi(:, wi), 1);
        for k = 1:K
            md(k, :) = accumarray(di, ci' .* mu(k, :));
        end
    end
    for k = 1:K
        mw(k, :) = accumarray(wi, ci' .* mu(k, :));
    end
    theta = normalise(md + ALPHA, 1);
    phi = normalise(mw + BETA, 2);
    perp = exp(sum(-log(sum(theta(:, di) .* phi(:, wi), 1)) .* ci')/sum(ci));
end
