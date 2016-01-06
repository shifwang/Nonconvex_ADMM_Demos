% An example of LqBasisPursuit
% Generate data
seed = 100;
rng(seed);
dim = 200;
sparsity = 40;
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
estone = LqBasisPursuit(A,y,.5, 10);
%esthalf = LqBasisPursuit(A,y,.5,100)
%FIXME: two unsolved bugs: [1] L1 basis pursuit always exits at max iteration [2] L1/2's lagrangian is too big and error is too large.
% Plot result
%FIXME
