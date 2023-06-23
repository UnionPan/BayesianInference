function nn_output = neural_net_output(X,W_seq,beta)
% compute the output of a neural net 
[n_train, ~] = size(X);  % n_train: number of data row in training data
L = length(W_seq);

if L==1
    a = W_seq{1}*X';
    Z = [ones(1,n_train);activation(a,'ReLU')];
else
    a = W_seq{1}*X';
    Z = [ones(1,n_train);activation(a,'ReLU')];
    for j=2:L-1
        a = W_seq{j}*Z;
        Z = [ones(1,n_train);activation(a,'ReLU')];
    end
    a = W_seq{L}*Z;
    Z = [ones(1,n_train);activation(a,'ReLU')];
end
nn_output = Z'*beta;

end
