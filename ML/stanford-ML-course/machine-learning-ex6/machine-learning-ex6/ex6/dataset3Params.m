function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
#C = 1;
#sigma = 0.3;
C = 100;
sigma = 30;
% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
paramVec = [0.001 0.003 0.1 0.3 1 3 10 30];
val_error = 1000

for i=1:length(paramVec)
  for j=1:length(paramVec)
     fprintf("\n\n\n current-C = %f, current-sigma = %f \n", C, sigma);
     fprintf("Training-C = %f, Training-sigma = %f \n", paramVec(i), paramVec(j));
     model = svmTrain(X, y, paramVec(i), @(x1, x2) gaussianKernel(x1, x2, paramVec(j)));
     predictions = svmPredict(model, Xval);
     new_error = mean(double(predictions ~= yval));
     fprintf ("new_error = %f, val_error=%f \n", new_error, val_error);
     if new_error < val_error 
       fprintf ("new_error = %f is Less Than val_error=%f \n", new_error, val_error);
       pause;
       fprintf ("Updating C and Sigma from sigma = %f and C = %f ", sigma, C);
       pause;
       val_error = new_error;
       C = paramVec(i);
       sigma = paramVec(j);
       fprintf ("to sigma = %f and C = %f \n\n\n", sigma, C);
       pause;
     endif
  endfor
endfor

fprintf("final C and sigma are C = %f, sigma = %f", C, sigma);
pause;
% =========================================================================

end
