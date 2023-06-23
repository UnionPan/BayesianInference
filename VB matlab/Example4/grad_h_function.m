function f = grad_h_function(beta,X,y,sigma2_hp)

pi = 1./(1+exp(-X*beta));
grad_log_llh = X'*(y-pi);

grad_prior = -beta/sigma2_hp;

f = grad_prior+grad_log_llh;

end
    

