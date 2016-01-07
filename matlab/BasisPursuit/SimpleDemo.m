% An example of LqBasisPursuit
% Clear environment
clc, clear all, close all;
% Generate data
seed = 1009;
rng(seed);
dim = 200;
sparsity = 35;
measurement = 100;
scoreboard = zeros(2,2);
distribution = 'binary';
if strcmp(distribution,'binary')
  x = [sign(randn(sparsity,1));zeros(dim - sparsity,1)];
elseif strcmp(distribution,'Gaussian')
  x = [randn(sparsity,1);zeros(dim - sparsity,1)];
end 
x = x(randperm(length(x)));
A = randn(measurement, dim);
y = A * x;
% Run code
[est_one, FLAG_one, iter_one, beta_one, obj_one, err_one, Lagrangian_one, timing_one] = LqBasisPursuit(A,y,1, 10);
[est_half, FLAG_half, iter_half, beta_half, obj_half, err_half, Lagrangian_half, timing_half] = LqBasisPursuit(A,y,.5,100);
% Plot 1: L1 objective function
figure()
plot(obj_one)
title('L_1 basis pursuit')
xlabel('iteration')
ylabel('objective ||x||_1')
%Plot 2: L1 recovery result
figure()
stem([est_one,x])
title('L_1 recovery result')
xlabel('dimension')
ylabel('value')
legend('recovered','true')
% Plot 3: L_0.5 objective function
figure()
plot(obj_half)
title('L_{0.5} basis pursuit')
xlabel('iteration')
ylabel('objective ||x||_1')
%Plot 2: L_0.5 recovery result
figure()
stem([est_half,x])
title('L_{0.5} recovery result')
xlabel('dimension')
ylabel('value')
legend('recovered','true')