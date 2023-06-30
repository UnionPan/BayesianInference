function [f,TP_MCR,MCR] = prediction_loss(y_test,X_test,W_seq,beta)
% compute the prediction loss (minus log-likelihood), true-positive MCR and MCR for logistic-NN model

ntest = length(y_test);
nnet_output = neural_net_output(X_test,W_seq,beta);
f = sum(-y_test.*nnet_output+log(1+exp(nnet_output)));
f = f/ntest;

y_pred = nnet_output>0;
MCR = mean(abs(y_test-y_pred)); % missclassification rate

idx = find(y_test==1);
ytest_1 = y_test(idx);
y_pred_1 = y_pred(idx);

TP_MCR = mean(abs(ytest_1 - y_pred_1));% True positive MCR

end



