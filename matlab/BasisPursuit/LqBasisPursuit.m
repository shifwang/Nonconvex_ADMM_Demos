function [x, FLAG, iter, betas, obj, err, Lagrangian, timing, err_rel, err_abs] = LqBasisPursuit(A, y, q,beta)
  % Lq_basis_pursuit  Solve Lq basis pursuit via ADMM
  %
  % Solves the following problem via ADMM:
  %
  %   minimize     ||x||_q^q
  %   subject to   Ax = y
  %
  % Args:
  %   A         : matrix
  %   y         : observation vector
  %   q         : 0 < q <= 1 constant
  %
  % Returns:
  %   x         : estimated vector 
  %   FLAG      : a string indicating the exit status
  %              'Max iteration' : Max iteration has been met.
  %              'Relative error': RELTOL has been met, i.e., 
  %                    ||x^k - z^k||_inf < RELTOL * \|c(x^k,z^k)\|_inf
  %              'Absolute error': ABSTOL has been met, i.e.,
  %                    ||x^k - z^k||_inf < ABSTOL
  %              'Unbounded'     : sequence is unbounded, i.e.,
  %                    ||x^k||_inf > LARGE
  %              'beta too large': cannot find appropriate beta, i.e.,
  %                    beta > LARGE (only happens when beta is set automatically)
  %                   
  %   iter       : final iteration
  %   betas      : beta of different iterates
  %   obj        : vector of || x^k ||_q^q for each k
  %   err        : vector of || x^k - z^k ||_inf for each k
  %   Lagrangian : vector of Lagrangian function for each k
  %   timing     : elapsed time of the algorithm
  %   err_rel    : the relative lower bound of different iterates
  %   err_abs    : the absolute lower bound of different iterates
  %
  % More information can be found in the paper linked at:
  %   http://arxiv.org/abs/1511.06324
  %
  
  % other parameters
  %   x0        : initial point, default is the zero vector
  %   beta      : penalty parameter, default is 10
  %   AUTO      : whether beta can be changed by the program
  %   MAXCOUNTS : when Lagrangian increases MAXCOUNTS times, beta = beta * SCALE
  %   SCALE     : beta = beta * SCALE when Lagrangian increases MAXCOUNTS times.
  %   RELTOL    : relative error tolerance 
  %   ABSTOL    : absolute error tolerance 
  %   MAXITER   : max iteration
  %   VERBOSE   : whether print details of the algorithm
  x0        = zeros(size(A,2),1);
  AUTO      = true;
  SCALE     = 1.2;
  RELTOL    = 1e-6;
  ABSTOL    = 1e-7;
  MAXCOUNTS = 100;
  MAXITER   = 1000;
  VERBOSE   = true;
  % Sanity check
  assert(size(A,1) == size(y,1));
  assert(size(y,2) == 1);
  assert(q > 0 & q <= 1);
  assert(rank(A) == size(A,1));
  assert(beta > 0);
  assert(size(A,2) == size(x0,1));
  assert(size(x0,2) == 1);
  % Default constant
  LARGE  = 1e6;
  [m, n] = size(A);
  if AUTO
    increasecounts = 0;
  end
  % Main body
  tic; % record the time
  % Initialize
  x = x0;
  z = x0;
  w = zeros(size(x0));
  obj     = nan(MAXITER, 1);
  err     = nan(MAXITER, 1);
  err_rel = nan(MAXITER, 1);
  err_abs = nan(MAXITER, 1);
  betas   = nan(MAXITER, 1);
  lagrng  = nan(MAXITER, 1);
  % pre-defined functions
  InfNorm    = @(x) max(abs(x));
  Lq         = @(x) sum(abs(x).^q);
  Lagrangian = @(x,z,w,beta) Lq(z) + w' * (x - z) + beta/2 * sum((x - z).^2);
  if VERBOSE
    fprintf('%4s\t%10s\t%10s\t%10s\n', 'iter', 'obj','lagrng', 'error');
  end
  % Calculate the projection x -> Px + q
  AAt = A * A';
  P = diag(ones(n,1)) - A' * (AAt\A);
  r = A' * (AAt\y);
  for k = 1:MAXITER 
    % x update
    %------------------------------------------------
    %   minimize   \| x - z + w/beta \|_2^2
    %   subject to Ax = y
    %------------------------------------------------
    x = P * (z - w/beta) + r;
    % z update
    %------------------------------------------------
    %   minimize_z \|z\|_q^q + beta/2 \| x - z + w/beta \|_2^2
    %------------------------------------------------
    z = Threshold(x + w/beta, beta,q);
    % w update
    w = w + beta * (x - z);
    % record the values
    obj(k)     = Lq(x);
    err(k)     = InfNorm(x - z);
    betas(k)   = beta;
    lagrng(k)  = Lagrangian(x, z, w, beta);
    err_rel(k) = RELTOL * InfNorm([x;z]);
    err_abs(k) = ABSTOL;
    if VERBOSE 
      fprintf('%4d\t%10.4f\t%10.4f\t%10.4f\n', k, obj(k), lagrng(k), err(k));
    end
    % beta update
    if AUTO && k > 1 && lagrng(k) > lagrng(k - 1);
      increasecounts = increasecounts + 1;
    end
    if AUTO && increasecounts > MAXCOUNTS;
      increasecounts = -1;
      beta = beta * SCALE;
    end
    % stopping criteria
    if k == MAXITER
      FLAG = 'Max iteration';
      break;
    end
    if AUTO && beta > LARGE
      FLAG = 'beta too large';
      break;
    end
    if InfNorm(x) > LARGE
      FLAG = 'Unbounded';
      break;
    end
    if err(k) < ABSTOL
      FLAG = 'Absolute error';
      break;
    end
    if err(k) < RELTOL * InfNorm([x;z])
      FLAG = 'Relative error';
      break;
    end
  end
  iter = k;
  timing = toc;
  if VERBOSE
    fprintf('ADMM has stopped at iter %4d because of %10s.\n',k,FLAG);
    fprintf('Elapsed time is %8.1e seconds .\n',timing);
  end
end
function z = HalfThres(x, beta)
  tmp = 3/2*(beta)^(-2/3);  
  z = 2/3*x.*(1+cos(2/3*pi - 2/3 * acos(1/(4*beta)*(abs(x)/3).^(-3/2))));
  z(abs(x) <= tmp) = 0;
end  
function out = Threshold(x, beta, q)
    %  solve argmin_z |z|_q^q + beta/2 | z - x |^2
    if q == 1
      out = ((x - 1/beta).*(x > 1/beta) + (x + 1/beta).*(x < -1/beta));
    elseif q == 1/2
      out = HalfThres(x,beta);
    else
      error('q can only be 1 or 0.5');
    end
end
