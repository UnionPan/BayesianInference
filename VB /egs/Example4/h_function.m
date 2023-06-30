function f = h_function(beta,X,y,sigma2_hp)
[~,d] = size(X);
log_prior = -d/2*log(2*pi)-d/2*log(sigma2_hp)-beta'*beta/2/sigma2_hp;
aux = X*beta;
log_llh = y'*aux-sum(aux+log(1+exp(-aux)));
f = log_prior+log_llh;

end
    

