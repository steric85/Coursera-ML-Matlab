function [theta] = normalEqn(X, y)
%NORMALEQN Computes the closed-form solution to linear regression 
%   NORMALEQN(X,y) computes the closed-form solution to linear 
%   regression using the normal equations.

theta = zeros(size(X, 2), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the code to compute the closed form solution
%               to linear regression and put the result in theta.
%

% ---------------------- Sample Solution ----------------------

X_squared = X' * X; % n+1 x n+1
X_squared_inverse = pinv(X_squared); % n+1 x n+1
X_transpose_y = X' * y; % n+1 x 1
theta = X_squared_inverse * X_transpose_y; % n+1 x 1

% -------------------------------------------------------------


% ============================================================

end
