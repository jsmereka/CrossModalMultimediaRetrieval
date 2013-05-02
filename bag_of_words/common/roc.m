function [ P, OP, A, T ] = roc ( X )

%  function [ P, OP, T ] = roc ( X )
%
%  Input data:
%
%  	X	N x 2	The first column contains the score, the second column contains 
%			the labels. Noise is labeled with 0, signal is labeled with k > 0.
%  Output data:
%
%       P       2 x N+1   x and y coords of curve...
%       OP      error at equal error rate (intersection with curve diagonal)
%       A       area under curve 
%       T       threshold at equal error

% M. Weber, California Institute of Technology, 2000.
  
  
N_SAMPLES = size(X, 1);
N_SIGNAL  = length(find(X(:, 2)));
N_NOISE   = N_SAMPLES - N_SIGNAL;


[D, I] = sort(rand(N_SAMPLES, 1));
X = X(I, :);
[D, I] = sort(X(:, 1));
X = flipud(X(I, :));

P = cumsum([ (X(:, 2) > 0) / N_SIGNAL, (X(:, 2) == 0) / N_NOISE ]);

% Compute area under curve
noiseIdx = find(~X(:, 2));
A = sum(P(noiseIdx, 1)) / N_NOISE;

P = fliplr(P);

% Find intersection point with diagonal
[dum idx] = min(abs(P(:, 1) - (1 - P(:, 2))));
OP = P(idx, 2);

% Get threshold
T = 0.5 * (X(idx, 1) + X(max(1, idx + 1), 1));






