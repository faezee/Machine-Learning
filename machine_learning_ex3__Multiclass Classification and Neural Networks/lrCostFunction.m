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


z= X*theta; % m*(n) * n*1 -->> m*1
SIG= sigmoid(z);

sigma_elements= -y.*(log(SIG)) - (1.-y).*(log(1-SIG));
part_1= sum(sigma_elements)/m;

theta_decreas= theta(2:size(theta,2));
reg_part= lambda*sum(theta.^2)/(2*m);


J= part_1 + reg_part;

%% ====================================================  GRAD! :D 

z= X*theta;	% m*(n+1) * (n+1)*1 -->> m*1
SIG= sigmoid(z);


sigma_expr= (SIG-y).*X; %  (m+1).*(m*(n+1))  m*(n+1)
part1= sum(sigma_expr,1)./m;

part_regularized= (lambda.*theta)./m;


grad= part1' + part_regularized;
grad(1)= part1(1);
	
% =============================================================

grad = grad(:);

end
