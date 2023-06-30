function [W_seq,beta,shrinkage_gamma_seq,Loss_DL] = DL_training(X_train,y_train,X_validation,y_validation,n_units,batchsize,eps0,isotropic)
% Traing a fGLM model with binary reponse y.
% Bayesian Adaptive Group Lasso is used on the first-layer weights; no
% regularization is put on the rest. 
% Reference: Tran, M., Nguyen, N., Nott, D., Kohn, R. (2020). Bayesian Deep Net GLM and GLMM. Journal of Computational and Graphical Statistics, 2020.
% INPUT
%   X_train, y_train:           training data (binary response)
%   X_validation, y_validation: validation data
%   n_units:                    vector specifying the numbers of units in
%                               each hidden layer
%   batchsize:                  mini-batch size used in each iteration
%   eps0:                       constant learning rate
%   isotropic:                  true if isotropic structure Sigma=bb'+c^2I is
%                               used, otherwise rank-1 structure Sigma=bb'+\diag(c.^2) is used
% OUTPUT
%   W_seq:                      the optimal weights upto the last hidden
%                               layer
%   beta                        the optimal weights that connect last hidden layer to the output
%   shrinkage_gamma_seq         update of shrinkage parameters over
%                               iteration
%   Loss_DL                     validation loss over iterations
% Written by: Minh-Ngoc Tran and Nghia Nguyen
% 
% =====================================================================%
data = [y_train,X_train];
datasize = length(y_train);

L = length(n_units); % the number of hidden layers
p = size(X_train,2)-1; % number of covariates
W_seq = cell(1,L); % cells to store weight matrices
index_track = zeros(1,L); % keep track of indices of Wj matrices: index_track(1) is the total elements in W1, index_track(2) is the total elements in W1 & W2,...
index_track(1) = n_units(1)*(p+1); % size of W1 is m1 x (p+1) with m1 number of units in the 1st hidden layer 
W1_tilde_index = n_units(1)+1:index_track(1); % index of W1 without biases, as the first column if W1 are biases
for j = 2:L
    index_track(j) = index_track(j-1)+n_units(j)*(n_units(j-1)+1);
end
d_w = index_track(L); % the total number of weights up to (and including) the last layer
d_beta = n_units(L)+1; % dimension of the weights beta connecting the last layer to the output
d_theta = d_w+d_beta; % the total number of parameters
%--------------------------------------
% initialise the weights
    layers = [size(X_train,2) n_units 1];
    weights = InitializeNN(layers);
    mu=[];
    for i=1:length(layers)-1
        mu=[mu;weights{i}(:)];
    end
%----------------------------------------
b = normrnd(0,0.01,d_theta,1); % initialise b and c
if isotropic 
    c = .01;
else
    c = .01*ones(d_theta,1);
end
lambda=[mu;b;c];

S = 10; % the number of Monte Carlo samples to estimate the gradient
tau = 10000; % threshold tau before reducing constant learning rate eps0
grad_weight = .6; %weight in the momentum 

W1 = reshape(mu(1:index_track(1)),n_units(1),p+1);
W_seq{1} = W1; 
for j = 2:L
    index = index_track(j-1)+1:index_track(j);
    Wj = reshape(mu(index),n_units(j),n_units(j-1)+1);
    W_seq{j} = Wj; 
end
beta = mu(d_w+1:d_theta);

[Loss_current,~,~] = prediction_loss(y_validation,X_validation,W_seq,beta); % compute prediction loss
Loss_current
Loss_DL(1) = Loss_current;

shrinkage_gamma = .01*ones(p,1); % initialise gamma_beta, the shrinkage parameters
mu_tau = zeros(p,1); lambda_tau = zeros(p,1); % parameters for the auxiliary tau_j
mu_matrixW1_tilde = reshape(mu(W1_tilde_index),n_units(1),p);
b_matrixW1_tilde = reshape(b(W1_tilde_index),n_units(1),p);
if isotropic
    for j = 1:p
        mean_column_j_tilde = mu_matrixW1_tilde(:,j)'*mu_matrixW1_tilde(:,j)+...
            b_matrixW1_tilde(:,j)'*b_matrixW1_tilde(:,j)+c^2*n_units(1);
        mu_tau(j) = shrinkage_gamma(j)/sqrt(mean_column_j_tilde);        
    end
    lambda_tau = shrinkage_gamma.^2;
else
    c_matrixW1_tilde = reshape(c(W1_tilde_index),n_units(1),p);
    for j = 1:p
        mean_column_j_tilde = mu_matrixW1_tilde(:,j)'*mu_matrixW1_tilde(:,j)+...
            b_matrixW1_tilde(:,j)'*b_matrixW1_tilde(:,j)+sum(c_matrixW1_tilde(:,j).^2);
        mu_tau(j) = shrinkage_gamma(j)/sqrt(mean_column_j_tilde);
    end
    lambda_tau = shrinkage_gamma.^2;
end
mean_inverse_tau = mu_tau; % VB mean <1/tau_j>
shrinkage_gamma_seq = shrinkage_gamma; %

minibatch = datasample(data,batchsize);
y = minibatch(:,1);
X = minibatch(:,2:end);

rqmc = normrnd(0,1,S,d_theta+1); 
for s=1:S
    U_normal = rqmc(s,:)';
    epsilon1=U_normal(1);
    epsilon2=U_normal(2:end);
    theta=mu+epsilon1*b+c.*epsilon2;   

    W_seq = cell(1,L);        
    W1 = reshape(theta(1:index_track(1)),n_units(1),p+1);
    W_seq{1} = W1;
    for j = 2:L
        index = index_track(j-1)+1:index_track(j);
        Wj = reshape(theta(index),n_units(j),n_units(j-1)+1);
        W_seq{j} = Wj; 
    end
    beta = theta(d_w+1:d_theta);    
    
    W1_tilde = W1(:,2:end); % weights without biases
    W1_tilde_gamma = W1_tilde*diag(mean_inverse_tau);
    grad_prior_w_beta = [zeros(n_units(1),1);-W1_tilde_gamma(:);zeros(d_w+d_beta-index_track(1),1)];        
        
    grad_llh = gradient_log_likelihood(W_seq,beta,X,y,datasize);
     
    grad_h = grad_prior_w_beta+grad_llh; % gradient of log prior plus log-likelihood
    grad_log_q = grad_log_q_function(b,c,theta,mu,isotropic);
    grad_theta = grad_h-grad_log_q;
    
    grad_g_lik_store(s,:) = [grad_theta;epsilon1*grad_theta;epsilon2.*grad_theta]';
end
grad_lb = (mean(grad_g_lik_store))';

gradient_lambda = inverse_fisher_times_grad(b,c,grad_lb,isotropic);
gradient_bar = gradient_lambda;

max_iter=100000;
iter=1;
stop=false;
patience_parameter = 50; % stop if test error not improved after patience_parameter iterations
lambda_best = lambda; 
patience = 0;

while ~stop 
    iter = iter+1
    
    minibatch = datasample(data,batchsize);
    y = minibatch(:,1);
    X = minibatch(:,2:end);
    rqmc = normrnd(0,1,S,d_theta+1);
    for s=1:S
        U_normal = rqmc(s,:)';
        epsilon1=U_normal(1);
        epsilon2=U_normal(2:end);
        theta=mu+b*epsilon1+c.*epsilon2;   

        W_seq = cell(1,L);        
        W1 = reshape(theta(1:index_track(1)),n_units(1),p+1);
        W_seq{1} = W1;
        for j = 2:L
            index = index_track(j-1)+1:index_track(j);
            Wj = reshape(theta(index),n_units(j),n_units(j-1)+1);
            W_seq{j} = Wj; 
        end
        beta = theta(d_w+1:d_w+d_beta);    

        W1_tilde = W1(:,2:end); % weights without biases
        W1_tilde_gamma = W1_tilde*diag(mean_inverse_tau);
        grad_prior_w_beta = [zeros(n_units(1),1);-W1_tilde_gamma(:);zeros(d_w+d_beta-index_track(1),1)];        

        grad_llh = gradient_log_likelihood(W_seq,beta,X,y,datasize);

        grad_h = grad_prior_w_beta+grad_llh;
        grad_log_q = grad_log_q_function(b,c,theta,mu,isotropic);
        grad_theta = grad_h-grad_log_q;
    
        grad_g_lik_store(s,:) = [grad_theta;epsilon1*grad_theta;epsilon2.*grad_theta]';
    end
    grad_lb = (mean(grad_g_lik_store))';
    gradient_lambda = inverse_fisher_times_grad(b,c,grad_lb,isotropic);
    
    grad_norm = norm(gradient_lambda);
    norm_gradient_threshold = 100;
    if norm(gradient_lambda)>norm_gradient_threshold
        gradient_lambda = (norm_gradient_threshold/grad_norm)*gradient_lambda;
    end
    
    gradient_bar_old = gradient_bar;
    gradient_bar = grad_weight*gradient_bar+(1-grad_weight)*gradient_lambda;     
    
    if iter>tau
        stepsize=eps0*tau/iter;
    else
        stepsize=eps0;
    end
    
    lambda=lambda+stepsize*gradient_bar;
    
    mu=lambda(1:d_theta,1);
    b=lambda(d_theta+1:2*d_theta,1);
    c=lambda(2*d_theta+1:end);

    W1 = reshape(mu(1:index_track(1)),n_units(1),p+1);
    W_seq{1} = W1; 
    for j = 2:L
        index = index_track(j-1)+1:index_track(j);
        Wj = reshape(mu(index),n_units(j),n_units(j-1)+1);
        W_seq{j} = Wj; 
    end
    beta = mu(d_w+1:d_theta);

    % update tau and shrinkage parameters    
    if mod(iter,10) == 0
        mu_matrixW1_tilde = reshape(mu(W1_tilde_index),n_units(1),p);
        b_matrixW1_tilde = reshape(b(W1_tilde_index),n_units(1),p);
        if isotropic
            for j = 1:p
                mean_column_j_tilde = mu_matrixW1_tilde(:,j)'*mu_matrixW1_tilde(:,j)+...
                    b_matrixW1_tilde(:,j)'*b_matrixW1_tilde(:,j)+c^2*n_units(1);
                mu_tau(j) = shrinkage_gamma(j)/sqrt(mean_column_j_tilde);
                lambda_tau(j) = shrinkage_gamma(j)^2;
            end
        else
            c_matrixW1_tilde = reshape(c(W1_tilde_index),n_units(1),p);
            for j = 1:p
                mean_column_j_tilde = mu_matrixW1_tilde(:,j)'*mu_matrixW1_tilde(:,j)+...
                    b_matrixW1_tilde(:,j)'*b_matrixW1_tilde(:,j)+sum(c_matrixW1_tilde(:,j).^2);
                mu_tau(j) = shrinkage_gamma(j)/sqrt(mean_column_j_tilde);
                lambda_tau(j) = shrinkage_gamma(j)^2;
            end
        end
        mean_inverse_tau = mu_tau;
        mean_tau = 1./mu_tau+1./lambda_tau;
        shrinkage_gamma = sqrt((n_units(1)+1)./mean_tau);
        shrinkage_gamma_seq = [shrinkage_gamma_seq,shrinkage_gamma];
    end
    
    
    [Loss_current,~,~] = prediction_loss(y_validation,X_validation,W_seq,beta); % compute prediction loss
    Loss_current
    Loss_DL(iter) = Loss_current;

    if Loss_DL(iter)>=Loss_DL(iter-1)
        gradient_bar = gradient_bar_old;
    end
    
    if Loss_DL(iter)<=min(Loss_DL)
        lambda_best = lambda;
        patience = 0;
    else
        patience = patience+1;
    end
    
    if (patience>patience_parameter)||(iter>max_iter) stop = true; end 
end
plot(Loss_DL);

lambda = lambda_best;
mu=lambda(1:d_theta,1);
W1 = reshape(mu(1:index_track(1)),n_units(1),p+1);
W_seq{1} = W1; 
for j = 2:L
    index = index_track(j-1)+1:index_track(j);
    Wj = reshape(mu(index),n_units(j),n_units(j-1)+1);
    W_seq{j} = Wj; 
end
beta = mu(d_w+1:d_w+d_beta);

end