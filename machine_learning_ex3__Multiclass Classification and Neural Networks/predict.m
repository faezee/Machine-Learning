function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%


X = [ones(m, 1) X];			% Add ones to the X 


z_2 = Theta1 * X'; 			% [second_layer_units*(m+1)] * [(m+1)*n]  -->> second_layer_units*n
a_2 = sigmoid(z_2);

a_2 = [ones(1, size(a_2, 2)); a_2];			% Add ones to the a_2


z_3 = Theta2 * a_2; 		% [third_layer_unit*second_layer_units] * [second_layer_units*n] == third_layer_unit*n
a_3 = sigmoid(z_3);

% fprintf('size of a_3, output\n');
% size(a_3)

% p is the indexes of max_value, so is the class of that input
[max_value, p]= max(a_3', [], 2);



% =========================================================================


end
