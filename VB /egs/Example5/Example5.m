% Implement Gaussian VB with factor decomposition
% for the deep neural network example
% Reference: "A practical tutorial on Variational Bayes" by Minh-Ngoc Tran 
clear all
load('data_census.mat'); % note that the data have been standardized
rng(1000)% fixed randon seed

batchsize = 1000; % size of mini-matches in training 
n_units = [100,100]; % hidden layers 
eps0 = .01; % fixed learning rate
isotropic = true; % 

[W_seq,beta,shrinkage_gamma_seq,Loss_DL] = DL_training(X,y,X_validation,y_validation,n_units,batchsize,eps0,isotropic);

fontsize = 20;
plot(Loss_DL)
xlabel('Iterations','FontSize', fontsize)
ylabel('Validation loss','FontSize', fontsize)

