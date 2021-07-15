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

hypothesis = X*theta;            %Hypothesis for all members of Training Set in a m dimention vecotor
temp = sum((hypothesis - y).^2); %Hypothesis and y are in the same dimention. 
sigma = sum(temp);               %temp is a m dimention vector, and sigma create a scalar 
legacyJ = (1/(2*m)) * sigma; 
regularizationParam = (lambda /(2*m)) .* ((sum(theta .^ 2))-(theta(1)^2));
J = legacyJ + regularizationParam;

%Gradient Descent
%dim(sig)=m*1; dim(y)=m*1  => dime(temp)=m*1
temp = hypothesis - y;                     

% dim(temp')=1*m; dim(X)=m*n; => dim(legacyGrad)=n*1
legacyGrad = (1/m) * ((temp)' * X)';

% dim(regularizationParam)= dim(theta)= n*1
regularizationParam = (lambda/m) * theta; 
regularizationParam (1) = regularizationParam(1)- (lambda/m) * theta(1);

% dim(grad)= dim(theta)= dim(legacyGrad)= n*1
grad = legacyGrad + regularizationParam;
% =========================================================================

grad = grad(:);

end
