function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
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
%
% Hint: The computation of the cost function and gradients can be
%       efficiently vectorized. For example, consider the computation
%
%           sigmoid(X * theta)
%
%       Each row of the resulting matrix will contain the value of the
%       prediction for that example. You can make use of this to vectorize
%       the cost function and gradient computations. 
%
% Hint: When computing the gradient of the regularized cost function, 
%       there're many possible vectorized solutions, but one solution
%       looks like:
%           grad = (unregularized gradient for logistic regression)
%           temp = theta; 
%           temp(1) = 0;   % because we don't add anything for j = 0  
%           grad = grad + YOUR_CODE_HERE (using the temp variable)
%



% Hypothesis of Logistic Regrssion which is a Sigmoid Function
% dim(X)=m*n; dim(theta)=n*1; => dim(h)=dime(sig)=m*1
sig=sigmoid(X*theta);

%Cost Function
term1=(-y).*log(sig);
term2=(ones(m,1)-y).* log(ones(m,1)-sig);
legacyJ = 1/m *(sum(term1-term2));
#regularizationParam = (lambda /(2*m)) .* ((sum(theta .^ 2))-(theta(1)^2));
regularizationParam = (lambda /(2*m)) .* sum(theta(2:end) .^ 2);
J = regularizedJ= legacyJ + regularizationParam;

%Gradient Descent

%dim(sig)=m*1; dim(y)=m*1  => dime(temp)=m*1
temp = sig - y;                     

% dim(temp')=1*m; dim(X)=m*n; => dim(legacyGrad)=n*1
#legacyGrad = (1/m) * ((temp)' * X)';
legacyGrad = (1/m) * (X'* temp);

% dim(regularizationParam)= dim(theta)= n*1
#regularizationParam = (lambda/m) * theta; 
#regularizationParam (1) = regularizationParam(1)- (lambda/m) * theta(1);
regularizationParam = (lambda/m) * theta(2:end,:); 
regularizationParam = [0; regularizationParam];

% dim(grad)= dim(theta)= dim(legacyGrad)= n*1
grad = regularizedGrad = legacyGrad + regularizationParam;
% =============================================================

grad = grad(:);

end
