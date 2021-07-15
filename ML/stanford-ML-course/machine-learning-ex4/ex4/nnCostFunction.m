function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%
% -------------------------------------------------------------

%Forward Propagation
%
% The algorithm works with calculated propabilities in hx, not with binary 
% results. So, you don't need to find max and set it to 1, and the others to 0. 
%

X=[ones(m,1) X]; %Bias Unit
a1 = X';

z2=Theta1*a1;  
a2=sigmoid(z2);
a2 = [ones(1,size(a2,2)); a2]; %Bias Unit. size(a2)=>10*5000 

z3 = Theta2*a2; 
hx = a3 = sigmoid(z3);  

%Cost Function
Y=zeros(size(y),num_labels);
for i=1:num_labels
  pos = find (y == i);
  Y(pos,i) = 1;
endfor

loghx = log(hx); loghx_=log (1-hx); cost_i = 0;
for i=1:m
   cost_i = cost_i  + Y(i,:)*loghx(:,i) + (1-Y(i,:))*loghx_(:,i);
endfor
J = J - 1/m *(cost_i);

% Regularization
regParam = (lambda /(2*m)) .* sum(nn_params .^ 2);
regParam  = regParam - (lambda /(2*m)) .* (( ...            % Omit bias effect 
                                             sum(Theta1(:,1).^2) + ...
                                             sum(Theta2(:,1).^2) ...
                                           ));              
J = J + regParam;


% Backppropagation

capDelta_L1 = zeros(size (Theta1)); 
capDelta_L2 = zeros(size (Theta2));

a1 = X';
delta_3 = a3 - Y';
delta_2 = (Theta2'* delta_3)(2:end, :) .* sigmoidGradient(z2);

% Gradient Value
capDelta_L2 = capDelta_L2 + delta_3 * a2';
capDelta_L1 = capDelta_L1 + delta_2 * a1';

%Regularization
Theta1_grad = 1/m .* capDelta_L1 + lambda/m .* Theta1; 
Theta1_grad(:,1) = Theta1_grad(:,1) - lambda/m .* Theta1(:,1) ; % Omit bias effect
Theta2_grad = 1/m .* capDelta_L2 + lambda/m .* Theta2;
Theta2_grad(:,1) = Theta2_grad(:,1) - lambda/m .* Theta2(:,1) ; % Omit bias effect

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end