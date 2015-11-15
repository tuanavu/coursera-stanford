function [X_norm, mu, sigma] = featureNormalize(X)
%FEATURENORMALIZE Normalizes the features in X 
%   FEATURENORMALIZE(X) returns a normalized version of X where
%   the mean value of each feature is 0 and the standard deviation
%   is 1. This is often a good preprocessing step to do when
%   working with learning algorithms.

% You need to set these values correctly
X_norm = X;
mu = zeros(1, size(X, 2));
sigma = zeros(1, size(X, 2));

% ====================== YOUR CODE HERE ======================
% Instructions: First, for each feature dimension, compute the mean
%               of the feature and subtract it from the dataset,
%               storing the mean value in mu. Next, compute the 
%               standard deviation of each feature and divide
%               each feature by it's standard deviation, storing
%               the standard deviation in sigma. 
%
%               Note that X is a matrix where each column is a 
%               feature and each row is an example. You need 
%               to perform the normalization separately for 
%               each feature. 
%
% Hint: You might find the 'mean' and 'std' functions useful.
%       

% Method 1: Using for-loop
%{
mu = mean(X); % mean of the feature
sigma = std(X); % standard deviation
for i = 1:size(X,2)
    X_norm(:,i) = (X(:,i) - mu(i)) / sigma(i);
end
%}


% Method 2: Without for-loop
mu = mean(X);              % returns a row vector
sigma = std(X);            % returns a row vector
m = size(X, 1);            % returns the number of rows in X
mu_matrix = ones(m, 1) * mu; 
sigma_matrix = ones(m, 1) * sigma;

% subtract the mu matrix from X, and divide element-wise by the sigma matrix, 
% and arrive at X_normalized
X_norm = (X - mu_matrix) ./ sigma_matrix; 

% ============================================================

end
