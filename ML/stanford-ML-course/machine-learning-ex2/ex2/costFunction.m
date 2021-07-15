function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

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
%
% Note: grad should have the same dimensions as theta

% Hypothesis of Logistic Regrssion which is a Sigmoid Function
h=X*theta;                   % dim(X)=m*n; dim(theta)=n*1; => dim(h)=m*1
sig=sigmoid(h);              % dim(h)=dime(sig)=m*1

%Cost Function
term1=(-y).*log(sig);
term2=(ones(m,1)-y).* log(ones(m,1)-sig);
summation= sum(term1-term2);
J= 1/m *(summation);

%Gradient Descent
temp = sig - y;                 % dim(sig)=m*1; dim(y)=m*1    => dime(temp)=m*1
Sigma = temp' * X;              % dim(temp')=1*m; dim(X)=m*n; => dim(sigma)= 1*n
grad = (1/m) * Sigma';  %dim(theta)=dime(sigma')=n*1

% =============================================================

end
