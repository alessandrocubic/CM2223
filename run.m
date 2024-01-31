clear; clear global;

lambda = 1e0
[X_hat, y_hat, w_star] = construct_matrix("./ML-CUP22-TR.csv", lambda);
[f_lls, grad_lls] = build_lls(X_hat, y_hat);

% Compute the solution using L-BFGS
w = zeros(size(w_star)); % starting point
b = X_hat' * y_hat;
max_iters = 1e3;

global metrics;
metrics.rel_errors = -ones(max_iters, 1);
metrics.residual = -ones(max_iters, 1);

[w_our, k, w_hist] = LBFGS(w, sparse(X_hat), b, 5, 1e-14, true, max_iters,...
    @(x,k) metrics_cb(x, k, w_star, X_hat, y_hat));

% Compute the relative errors and the final residual
% rel_errors = vecnorm(w_hist - w_star) / norm(w_star);
% residual = norm(X_hat * w_hist(:, end) - y_hat) / norm(y_hat);