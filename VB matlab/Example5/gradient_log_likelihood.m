function grad_llh = gradient_log_likelihood(W_seq,beta,X,y,datasize)
% compute the gradient of log-likelihiood for logistic-NN model
% n = size(X,1);
% grad_llh = 0;
% for i = 1:n
%     yi = y(i);
%     xi = X(i,:); xi = xi';
%     nnet_output = neural_net_output(xi,W_seq,beta);
%     gradient = nn_backpropagation(xi,W_seq,beta);
%     grad_llh = grad_llh+(yi-1/(1+exp(-nnet_output)))*gradient;
% end

n = length(y);
% node_store = neural_net_output(X,W_seq,beta);
% nn_output = node_store{end};
grad_llh = datasize/n*nn_backpropagation(X,y,W_seq,beta);
% gradient_w_beta = exp(-theta_sigma2)*back_prop;
% gradient_theta_sigma2 = -1/2*n+1/2*exp(-theta_sigma2)*sum((y-nn_output').^2);
% grad_llh = [gradient_w_beta;gradient_theta_sigma2];

end
