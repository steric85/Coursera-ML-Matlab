function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

oneByM = 1/m;       %scalar

XintoTheta = X * theta;     % m x 1 

h = sigmoid(XintoTheta);    % m x 1 

theta1ToN = theta(2:size(theta));       % n x 1

% Calculating cost

lFirstTerm = y .* log(h);    % m x 1

lSecondTerm = (1 - y) .* log(1 - h);     % m x 1

leftTerm = -oneByM * sum(lFirstTerm + lSecondTerm);      %scalar

rightTerm = 0.5 * lambda * oneByM * (theta1ToN' * theta1ToN);   % scalar

J = leftTerm + rightTerm;

% Calculating gradient

HminusY = h - y;            % m x 1

leftTerm = oneByM * (X' * HminusY);         % (n+1) x 1

rightTerm = [0; lambda * oneByM * theta1ToN];       % (n+1) x 1

grad = leftTerm + rightTerm;        % (n+1) x 1

% =============================================================

end
