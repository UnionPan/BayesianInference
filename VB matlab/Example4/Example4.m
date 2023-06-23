% Implement Gaussian VB with the Cholesky decomposition
% for the logistic regression example
% Reference: "A practical tutorial on Variational Bayes" by Minh-Ngoc Tran 

clear all
rng(2020) 
data = xlsread('MROZ.xlsx'); % load the data
y = data(:,1);
n = length(y);
X = [ones(n,1),data(:,2:end)];

% FFVB
d_theta = size(X,2); dim = d_theta+d_theta*(d_theta+1)/2;
S = 20;  % number of Monte Carlo samples
beta1_adap_weight = 0.6;
beta2_adap_weight = 0.6;
eps0 = 0.001;
patience_max = 10;
tau_threshold = 100;
t_w = 10;

% hyperparameter
sigma2_hp = 10; 

mu = glmfit(X,y,'binomial','constant','off'); 
L = 0.1*eye(d_theta);
lambda = [mu;vech(L)]; % initial lambda


rqmc = normrnd(0,1,S,d_theta); 
grad_LB_mu = 0;
grad_LB_L = 0;

for s = 1:S    
    varepsilon = rqmc(s,:)';
    beta = mu+L*varepsilon;
    
    grad_h = grad_h_function(beta,X,y,sigma2_hp);
    grad_LB_mu = grad_LB_mu+grad_h;
    grad_LB_L = grad_LB_L+vech(grad_h*(varepsilon'));    
end
grad_LB_mu = grad_LB_mu/S;
grad_LB_L = grad_LB_L/S+vech(diag(diag(L)));
grad_LB = [grad_LB_mu;grad_LB_L];

% gradient clipping
grad_norm = norm(grad_LB);
norm_gradient_threshold = 10;
if norm(grad_LB)>norm_gradient_threshold
    grad_LB = (norm_gradient_threshold/grad_norm)*grad_LB;
end

g_adaptive = grad_LB; v_adaptive = g_adaptive.^2; 
g_bar_adaptive = g_adaptive; v_bar_adaptive = v_adaptive; 

max_iter = 500;iter = 1;
stop = false;
LB = 0; LB_bar = 0; patience = 0;
while ~stop    
    iter
    
    mu = lambda(1:d_theta);
    L = vechinv(lambda(d_theta+1:end),2);
  
    rqmc = normrnd(0,1,S,d_theta); % using quasi MC random numbers      
    grad_LB_mu = 0;
    grad_LB_L = 0;
    
    LB_t = 0;
    for s = 1:S    
        % generate theta_s
        varepsilon = rqmc(s,:)';
        beta = mu+L*varepsilon;

        grad_h = grad_h_function(beta,X,y,sigma2_hp);
        grad_LB_mu = grad_LB_mu+grad_h;
        grad_LB_L = grad_LB_L+vech(grad_h*(varepsilon'));  
        
        LB_t = LB_t + h_function(beta,X,y,sigma2_hp);
    end
    grad_LB_mu = grad_LB_mu/S;
    grad_LB_L = grad_LB_L/S+vech(diag(diag(L)));
    grad_LB = [grad_LB_mu;grad_LB_L];
    
    % gradient clipping
    grad_norm = norm(grad_LB);
    norm_gradient_threshold = 10;
    if norm(grad_LB)>norm_gradient_threshold
        grad_LB = (norm_gradient_threshold/grad_norm)*grad_LB;
    end
    
    g_adaptive = grad_LB; v_adaptive = g_adaptive.^2; 
    g_bar_adaptive = beta1_adap_weight*g_bar_adaptive+(1-beta1_adap_weight)*g_adaptive;
    v_bar_adaptive = beta2_adap_weight*v_bar_adaptive+(1-beta2_adap_weight)*v_adaptive;
    
    if iter>=tau_threshold
        stepsize = eps0*tau_threshold/iter;
    else
        stepsize = eps0;
    end
    
    lambda = lambda+stepsize*g_bar_adaptive./sqrt(v_bar_adaptive);
    
    LB(iter) = LB_t/S+1/2*log(det(L*(L')))+d_theta/2;
    
    if iter>=t_w
        LB_bar(iter-t_w+1) = mean(LB(iter-t_w+1:iter));
        LB_bar(iter-t_w+1)
    end
       
    if (iter>t_w)&&(LB_bar(iter-t_w+1)>=max(LB_bar))
        lambda_best = lambda;
        patience = 0;
    else
        patience = patience+1;
    end
    
    if (patience>patience_max)||(iter>max_iter) stop = true; end 
        
    iter = iter+1;
 
end

lambda = lambda_best;
mu = lambda(1:d_theta);
L = vechinv(lambda(d_theta+1:end),2);
Sigma = L*(L');

for i = 1:8
    subplot(3,3,i)
    x = mu(i)-3*sqrt(Sigma(i,i)):0.001:mu(i)+3*sqrt(Sigma(i,i));
    y = normpdf(x,mu(i),sqrt(Sigma(i,i)));
    str = sprintf('theta_%d', i);
    plot(x,y,'-')
    title(str)
end
subplot(3,3,9)
plot(LB_bar)
title('Lower bound')





