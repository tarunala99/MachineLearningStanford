function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

temp1 = X * theta ;
temp2 = temp1 - y;
temp4 = temp2.*X;
temp4 = sum(temp4)/m;
length1 = size(theta,1);
temp5 = [0;ones(length1-1,1)];
temp5 = temp5.*theta;
temp5 = lambda * temp5 / m;
grad = temp4' + temp5;

temp2 = temp2.^2;
temp2 = sum(temp2);
temp2 = temp2 / (2*m);

temp3 = theta.^2;
temp3 = sum(temp3(2:end));
temp3 = temp3 * lambda / (2*m);
J = temp2 + temp3;

% =========================================================================

grad = grad(:);

end
