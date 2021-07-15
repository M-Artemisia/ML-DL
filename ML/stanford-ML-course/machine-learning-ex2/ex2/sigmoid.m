function g = sigmoid(z)
%SIGMOID Compute sigmoid function
%   g = SIGMOID(z) computes the sigmoid of z.

% You need to return the following variables correctly 
g = zeros(size(z));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the sigmoid of each value of z (z can be a matrix,
%               vector or scalar).

%For Real Numbers
%g = 1/(1+e^(-z))
%For Matrices Inputs
#minz=-z
#e=exp(minz)
#ep1=ones(size(z))+e
#g=1./ep1
g=1./ (ones(size(z))+exp(-z));

% =============================================================

end
