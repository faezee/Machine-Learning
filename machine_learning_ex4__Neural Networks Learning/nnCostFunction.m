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
%
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


% calculate the output first! :D  -- output_class
X= [ones(m, 1) X];		%	m*(n+1)

z_2= Theta1* X';		%	(hidden_layer_size*(n+1))*((n+1)*m) ==  hidden_layer_size*m
a_2= sigmoid(z_2);

a_2= [ones(1, size(a_2, 2)); a_2];			% Add ones to the a_2 --->> (hidden_layer_size+1)*(m)

z_3= Theta2*a_2;		%	(output_layer_size*(hidden_layer_size+1))*((hidden_layer_size+1)*(m)) = output_layer_size*m 
a_3= sigmoid(z_3);


% output_class is the indexes of max_value, so is the class of that input
% [max_value, output_class]= max(a_3', [], 2);				%%%%%% 

sigma_sum= 0;
for(m_index= 1:m)
	y_m= zeros(num_labels, 1);
	y_m(y(m_index))= 1;					%  this is the output vector for one of the examples - the m_index'th :D

	x_m= a_3(:,m_index);

	sigma_elements= -y_m.*(log(x_m))-(1-y_m).*(log(1-x_m));
	% if(m_index==1)
	% endif

	sigma_sum = sigma_sum + sum(sigma_elements);
endfor

J= sigma_sum/m;


% === adding regularization 
Theta1_reg= Theta1.^2;
Theta2_reg= Theta2.^2;

% delete the first column, since the bias is not in regularization! :D
regularization_part= lambda*(sum(Theta1_reg(1:end,2:end)(:)) + sum(Theta2_reg(1:end,2:end)(:)))/(2*m);

J= J + regularization_part;
% -------------------------------------------------------------


% =========================================================================
% =========================================================================


D_1 = zeros(size(Theta1));
D_2 = zeros(size(Theta2));


% using for structure(not matrix) , as in the experiment!
for(t= 1:m)

	% calculate the output layer values for example t
	a_1= X(t,:);		% number t example :D
	z_2= Theta1*a_1';
	a_2= sigmoid(z_2);
	a_2= [1 ; a_2];
	z_3= Theta2*a_2;
	a_3= sigmoid(z_3);	% the output !

	% output_layer_3= a_3(:,t) %% output layer has computed before, so just use the output for example t! :D

	% the traiting example vector for example t.
	y_t= zeros(num_labels,1);
	y_t(y(t))= 1;

	% calculate the delta for output layer
	delta_3= a_3 - y_t;

	% hidden layer
	z_2= [1 ; z_2];		%% TODO! :D
	delta_2= (Theta2'*delta_3).*sigmoidGradient(z_2);
	delta_2= delta_2(2:end);

	% accumulate the gradient
	D_2= D_2+ delta_3*a_2';
	D_1= D_1+ delta_2*a_1;

	if(t==1)
		% fprintf('size!\n')
		% size(delta_3*a_2')
		% size(Theta2_grad)
		% size(delta_2*a_1)
		% size(Theta1_grad)
	endif

endfor

Theta1_grad= D_1./m;
Theta2_grad= D_2./m;

%% adding regularization
add_for_reg_1= (lambda/m)*Theta1;
add_for_reg_2= (lambda/m)*Theta2;

% place zero for first column, since first column elements are for the bias.
add_for_reg_1(:,1)= zeros(size(Theta1,1), 1);	
add_for_reg_2(:,1)= zeros(size(Theta2,1), 1);	

Theta1_grad= Theta1_grad + add_for_reg_1;
Theta2_grad= Theta2_grad + add_for_reg_2;


% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
