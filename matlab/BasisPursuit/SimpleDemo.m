% An example of LqBasisPursuit
% Clear environment
clc, clear, close all;
% Generate data
seed = 100;
rng(seed);
dim = 200;
sparsity = 35;
measurement = 100;
scoreboard = zeros(2,2);
distribution = 'Gaussian';
if strcmp(distribution,'binary')
  x = [sign(randn(sparsity,1));zeros(dim - sparsity,1)];
elseif strcmp(distribution,'Gaussian')
  x = [randn(sparsity,1);zeros(dim - sparsity,1)];
end 
x = x(randperm(length(x)));
A = randn(measurement, dim);
y = A * x;
% Run code
[est_one, FLAG_one, iter_one, beta_one, obj_one, err_one, Lagrangian_one, timing_one, err_rel1, err_abs1] = LqBasisPursuit(A,y,1, 1);
[est_half, FLAG_half, iter_half, beta_half, obj_half, err_half, Lagrangian_half, timing_half, err_rel2, err_abs2] = LqBasisPursuit(A,y,.5,100);
% Plot 1: L1 objective function
figure();
plot(obj_one)
title('L_1 basis pursuit')
xlabel('iteration')
ylabel('objective ||x||_1')
%Plot 2: L1 recovery result
figure()
stem([x, est_one])
title('L_1 recovery result')
xlabel('dimension')
ylabel('value')
legend('true', 'recovered')
%Plot 3: L1 stopping criterion
figure()
MaxIter = length(err_one);
semilogy(1:MaxIter, err_one, ...
    1:MaxIter, err_rel1, '-.', ...
    1:MaxIter,err_abs1, '--');
title('L_1 stopping criterion')
xlabel('iteration')
ylabel('||x^k - z^k||_\infty')
legend('current','relative','absolute')
%Plot 4: L1 beta
figure()
plot(beta_one)
title('L_1 ADMM parameter \beta')
xlabel('iteration')
ylabel('\beta')


% Plot 5: L_0.5 objective function
figure();
plot(obj_half)
title('L_{0.5} basis pursuit')
xlabel('iteration')
ylabel('objective ||x||_{0.5}')
% Plot 6: L_0.5 recovery result
figure()
stem([x, est_half])
title('L_{0.5} recovery result')
xlabel('dimension')
ylabel('value')
legend('true', 'recovered')
% Plot 7: L_0.5 stopping criterion
figure()
MaxIter = length(err_half);
semilogy(1:MaxIter, err_half, ...
    1:MaxIter, err_rel2, '-.', ...
    1:MaxIter,err_abs2, '--');
title('L_{0.5} stopping criterion')
xlabel('iteration')
ylabel('||x^k - z^k||_\infty')
legend('current','relative','absolute')
% Plot 8: L_0.5 beta
figure()
plot(beta_half)
title('L_{0.5} ADMM parameter \beta')
xlabel('iteration')
ylabel('\beta')
