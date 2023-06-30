function grad_log_q = grad_log_q_function(b,c,theta,mu,isotropic)
x = theta-mu;
if isotropic
    grad_log_q = -x./c^2+(1/c^2)*((b'*x)/(c^2+(b'*b)))*b;
else   
    d = b./c.^2;
    grad_log_q = -x./c.^2+(d'*x)/(1+(d'*b))*d;
end

end
    
    



