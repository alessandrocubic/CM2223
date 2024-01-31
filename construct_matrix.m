function [X_hat, y_hat, w_star] = construct_matrix(dataset, lambda)
% Build the matrices needed for the methods to run
% Inputs:
%       dataset     a csv of a NxM matrix
%       lambda      
%
% Output:
%       X_hat       is a block matrix of size M+N x N where above there is
%                   the original NxM matrix transposed and a NxN identity 
%                   matrix multiplied by lambda below.
%       y_hat       array of expected values, where the upper part is a
%                   random array and the lower one is filled with zeros
%       w           starting point
%       w_star      optimal solution


% Build the X matrix from the dataset
X = readtable(dataset);
X = table2array(X);
X = X(:, 2:end); %removing the id column

% Build X_hat and y_hat
[m, n0] = size(X);
X_hat = [X'; lambda.*eye(m)];
[m, n] = size(X_hat);
y_hat = [randn(n0, 1); zeros(m-n0, 1)];

% Retrieve the optimal solution
w_star = X_hat\y_hat;
end
