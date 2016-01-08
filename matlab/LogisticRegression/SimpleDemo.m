% A simple example for Lq regularized logistic regression 
% Edited from 
%   http://web.stanford.edu/~boyd/papers/admm/logreg-l1/logreg_example.html
% Generate problem data
clc, clear, close all;
seed = 1000;
rng(seed)

dimension      = 50;
train_set_size = 500;
test_set_size  = 1000;
w = (sprandn(dimension, 1, 0.1));  % N(0,1), 10% sparse
v = randn(1);            % random intercept

noise = false;
A_train             = sprandn(train_set_size, dimension, 0.1);
if noise
    train_label = sign(A_train*w + v + 0.1*randn(train_set_size,1)); % labels with noise
else
    train_label  = sign(A_train*w + v);
end
A_test              = sprandn(test_set_size, dimension, 10/dimension);
test_label   = sign(A_test*w + v);

% Solve problem
beta = 10;
mu = 0.0001;
q = 0.5;
[x, intercp, FLAG, iter, betas, obj, err, Lagrangian, timing, err_rel, err_abs] = LqLogReg(A_train, train_label, q, mu, beta, true);

% Text report:
fprintf('Weight correlation is %5.2f. \n',full(mean(x.*w) / sqrt(mean(x.*x)) / sqrt(mean(w .* w))));
fprintf('Training error is %5.2f%%. \n',sum(train_label ~= sign(A_train*x + intercp))/length(train_label)* 100.0);
fprintf('Test     error is %5.2f%%. \n',sum(test_label ~= sign(A_test*x + intercp))/length(test_label)* 100.0);

% Plot 1: objective function
figure();
plot(obj)
title('L_q regularized logistic regression')
xlabel('iteration')
ylabel('objective Logistic(x) + m\mu ||x||_q^q')
%Plot 2: recovery result
figure()
stem([w, x])
title('L_q recovery result')
xlabel('dimension')
ylabel('value')
legend('true', 'recovered')
%Plot 3: stopping criterion
figure()
MaxIter = length(err);
semilogy(1:MaxIter, err, ...
    1:MaxIter, err_rel, '-.', ...
    1:MaxIter,err_abs, '--');
title('L_q stopping criterion')
xlabel('iteration')
ylabel('||x^k - z^k||_\infty')
legend('current','relative','absolute')
%Plot 4: beta
figure()
plot(betas)
title('L_q ADMM parameter \beta')
xlabel('iteration')
ylabel('\beta')
