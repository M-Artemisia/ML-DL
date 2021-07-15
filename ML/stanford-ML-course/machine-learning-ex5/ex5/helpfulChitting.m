hypothesis = X*theta;            %Hypothesis for all members of Training Set in a m dimention vecotor
temp = sum((hypothesis - y).^2); %Hypothesis and y are in the same dimention. 
sigma = sum(temp);               %temp is a m dimention vector, and sigma create a scalar 
J = (1/(2*m)) * sigma; 


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




