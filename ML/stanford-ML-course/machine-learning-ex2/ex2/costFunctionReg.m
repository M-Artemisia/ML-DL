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

% Hypothesis of Logistic Regrssion which is a Sigmoid Function
% dim(X)=m*n; dim(theta)=n*1; => dim(h)=dime(sig)=m*1
sig=sigmoid(X*theta);


%Cost Function
term1=(-y).*log(sig);
term2=(ones(m,1)-y).* log(ones(m,1)-sig);
legacyJ = 1/m *(sum(term1-term2));
regularizationParam = (lambda /(2*m)) .* ((sum(theta .^ 2))-(theta(1)^2));
J = regularizedJ= legacyJ + regularizationParam;

%Gradient Descent

%dim(sig)=m*1; dim(y)=m*1  => dime(temp)=m*1
temp = sig - y;                     

% dim(temp')=1*m; dim(X)=m*n; => dim(legacyGrad)=n*1
legacyGrad = (1/m) * ((temp)' * X)';

% dim(regularizationParam)= dim(theta)= n*1
regularizationParam = (lambda/m) * theta; 
regularizationParam (1) = regularizationParam(1)- (lambda/m) * theta(1);

% dim(grad)= dim(theta)= dim(legacyGrad)= n*1
grad = regularizedGrad = legacyGrad + regularizationParam;
% =============================================================

end