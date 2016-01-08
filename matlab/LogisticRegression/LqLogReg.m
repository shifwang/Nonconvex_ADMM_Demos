function [x, intercp, FLAG, iter, betas, obj, err, Lagrangian, timing, err_rel, err_abs] = LqLogReg(A, b, q, mu, beta, offset)
  % LqLogReg  Solve Lq logistic regression via ADMM
  %
  % solves the following problem via ADMM:
  %
  %   minimize   sum( log(1 + exp(-b_i*(a_i'x + intercp)) ) + m*mu*||x||_q^q  
  %
  % where m is the number of samples
  %
  % Args:
  %   A         : feature matrix [a_1';...;a_m]
  %   b         : observation vector [b_1;...;b_m]
  %   q         : 0 < q <= 1 constant, can only be 0.5 or 1.
  %   mu        : balancing parameter
  %   beta      : initial penalty parameter in ADMM
  %   offset    : whether allow intercept to be nonzero
  %
  % Returns:
  %   x         : estimated vector 
  %   intercp   : intercept
  %   FLAG      : a string indicating the exit status
  %              'Max iteration' : Max iteration has been met.
  %              'Relative error': RELTOL has been met, i.e., 
  %                    ||x^k - z^k||_inf < RELTOL * \| [x^k;z^k] \|_inf
  %              'Absolute error': ABSTOL has been met, i.e.,
  %                    ||x^k - z^k||_inf < ABSTOL
  %              'Unbounded'     : sequence is unbounded, i.e.,
  %                    ||x^k||_inf > LARGE
  %              'beta too large': cannot find appropriate beta, i.e.,
  %                    beta > LARGE (only happens when beta is set automatically)
  %                   
  %   iter       : final iteration
  %   betas      : beta of different iterates
  %   obj        : vector of  sum( log(1 + exp(-b_i*(a_i'w^k)) ) + m*mu*||w^k||_q^q
  %   err        : vector of || x^k - z^k ||_inf for each k
  %   Lagrangian : vector of Lagrangian function for each k
  %   timing     : elapsed time of the algorithm
  %   err_rel    : the relative lower bound of different iterates
  %   err_abs    : the absolute lower bound of different iterates
  %
  % More information can be found in the paper linked at:
  %   http://arxiv.org/abs/1511.06324
  % The original convex version can be found in (Jan. 7, 2016)
  %   http://web.stanford.edu/~boyd/papers/admm/logreg-l1/logreg.html
  %
  
  % other parameters
  %   x0        : initial point, default is the zero vector
  %   AUTO      : whether beta can be changed by the program
  %   MAXCOUNTS : when Lagrangian increases MAXCOUNTS times, beta = beta * SCALE
  %   SCALE     : beta = beta * SCALE when Lagrangian increases MAXCOUNTS times.
  %   RELTOL    : relative error tolerance 
  %   ABSTOL    : absolute error tolerance 
  %   MAXITER   : max iteration
  %   VERBOSE   : whether print details of the algorithm
  if (offset)
      x0    = zeros(size(A,2)+1,1);
      A     = [b,A];%FIXME: mismatch
  else
      x0    = zeros(size(A,2),1);
  end
  AUTO      = true;
  SCALE     = 1.2;
  RELTOL    = 1e-5;
  ABSTOL    = 1e-5;
  MAXCOUNTS = 100;
  MAXITER   = 1000;
  VERBOSE   = true;
  % Sanity check
  assert(size(A,1) == size(b,1));
  assert(size(b,2) == 1);
  assert(q==0.5 || q==1);
  assert(beta > 0);
  assert(size(A,2) == size(x0,1));
  assert(size(x0,2) == 1);
  % Default constant
  LARGE  = 1e6;
  [m, n] = size(A);
  if AUTO
    increasecounts = 0;
  end
  C = -A;%FIXME: there is a mismatch 
  % Main body
  tic; % record the time
  % Initialize
  x       = x0;
  z       = x0;
  w       = zeros(size(x0));
  obj     = nan(MAXITER, 1);
  err     = nan(MAXITER, 1);
  err_rel = nan(MAXITER, 1);
  err_abs = nan(MAXITER, 1);
  betas   = nan(MAXITER, 1);
  lagrng  = nan(MAXITER, 1);
  % pre-defined functions
  InfNorm    = @(x) max(abs(x));
  Lq         = @(x) sum(abs(x).^q);
  Logistic   = @(x) sum(log(exp(C * x) + 1));
  Lagrangian = @(x,z,w,beta) Lq(z(2:end))*m*mu + Logistic(x) + w' * (x - z) + beta/2 * sum((x - z).^2);
  if VERBOSE
    fprintf('%4s\t%10s\t%10s\t%10s\n', 'iter', 'obj','lagrng', 'error');
  end
  for k = 1:MAXITER 
    % x update
    %------------------------------------------------
    %   minimize   sum(log(exp(C * x) + 1)) + beta/2 * \| x - z + w/beta \|_2^2
    %------------------------------------------------
    x = update_x(C, x, beta, z, w);
    % z update
    %------------------------------------------------
    %   minimize_z m*mu*\|z\|_q^q + beta/2 \| x - z + w/beta \|_2^2
    %------------------------------------------------
    z = x + w/beta;
    z(2:end) = Threshold(z(2:end), beta/m/mu,q);
    
    % w update
    w = w + beta * (x - z);
    % record the values
    obj(k)     = Lq(x(2:end))*m*mu + Logistic(x);
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
  if offset
      intercp = x(1);
      x = x(2:end);
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

function x = update_x(C, x, beta, z, w)
    % solve the x update
    %   minimize [ sum log(1 + exp(Cx)) + beta/2 * ||x - z^k + w^k/beta||^2 ]
    % via Newton's method; for a single subsystem only.
    alpha = 0.1;
    BETA  = 0.5;
    TOLERANCE = 1e-7;
    MAX_ITER = 50;
    [m n] = size(C);
    I = eye(n);
    f = @(x) sum(log(1 + exp(C*x))) + beta/2*norm(x - z + w/beta).^2;
    for iter = 1:MAX_ITER
        fx = f(x);
        g = C'*(exp(C*x)./(1 + exp(C*x))) + beta*(x - z + w/beta);
        H = C' * diag(exp(C*x)./(1 + exp(C*x)).^2) * C + beta*I;
        dx = -H\g;   % Newton step
        dfx = g'*dx; % Newton decrement
        if abs(dfx) < TOLERANCE
            break;
        end
        % backtracking
        t = 1;
        while f(x + t*dx) > fx + alpha*t*dfx
            t = BETA*t;
        end
        x = x + t*dx;
    end
end
