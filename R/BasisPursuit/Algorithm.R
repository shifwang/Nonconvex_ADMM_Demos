LqBasisPursuit <- function(A,y,q,
                           x0 = NULL,
                           beta = 10,
                           AUTO = TRUE,
                           MAXCOUNTS = 100,
                           SCALE = 1.2,
                           RELTOL = 1e-6,
                           ABSTOL = 1e-7,
                           MAXITER = 1000,
                           VERBOSE = T) {
  # Lq_basis_pursuit  Solve Lq basis pursuit via ADMM
  #
  # Solves the following problem via ADMM:
  #
  #   minimize     ||x||_q^q
  #   subject to   Ax = y
  #
  # Args:
  #   A         : matrix
  #   y         : observation vector
  #   q         : 0 < q <= 1 constant
  #   x0        : initial point, default is NULL (set by program)
  #   beta      : penalty parameter, default is 10
  #   AUTO      : whether beta can be changed by the program
  #   MAXCOUNTS : when Lagrangian increases MAXCOUNTS times, beta <- beta * SCALE
  #   SCALE     : beta <- beta * SCALE when Lagrangian increases MAXCOUNTS times.
  #   RELTOL    : relative error tolerance 
  #   ABSTOL    : absolute error tolerance 
  #   MAXITER   : max iteration
  #   VERBOSE   : whether print details of the algorithm
  #
  # Returns:
  #   x         : estimated vector 
  #   FLAG      : a string indicating the exit status
  #              'Max iteration' : Max iteration has been met.
  #              'Relative error': RELTOL has been met, i.e., 
  #                    ||x^k - z^k||_inf < RELTOL * \|c(x^k,z^k)\|_inf
  #              'Absolute error': ABSTOL has been met, i.e.,
  #                    ||x^k - z^k||_inf < ABSTOL
  #              'Unbounded'     : sequence is unbounded, i.e.,
  #                    ||x^k||_inf > LARGE
  #              'beta too large': cannot find appropriate beta, i.e.,
  #                    beta > LARGE (only happens when beta is set automatically)
  #                   
  #   iter       : final iteration
  #   beta       : final beta
  #   obj        : vector of || x^k ||_q^q for each k
  #   error      : vector of || x^k - z^k ||_inf for each k
  #   Lagrangian : vector of Lagrangian function for each k
  #   timing     : elapsed time of the algorithm
  #
  # More information can be found in the paper linked at:
  #   http://arxiv.org/abs/1511.06324
  #
  # Required packages
  require(Matrix)
  
  # Sanity check
  stopifnot(is.matrix(A))
  stopifnot(is.matrix(y))
  stopifnot(ncol(y) == 1)
  stopifnot(nrow(A) == nrow(y))
  stopifnot(rankMatrix(A) == nrow(A))
  stopifnot((q>0 && q <=1))
  stopifnot(beta > 0)
  if (!is.null(x0)){
    stopifnot(is.matrix(x0))
    stopifnot(nrow(x0) == ncol(A))
    stopifnot(ncol(x0) == 1)
  }
  
  # Default constant
  LARGE <- 1e6
  EPS <- 1e-10
  m <- nrow(A)
  n <- ncol(A)
  if (AUTO) {
    increase.counts <- 0
  }

  
  # Main body
  ptm <- proc.time() # record the time
  # Initialize
  if (is.null(x0)){
    x <- rep(0,n)
    z <- rep(0,n)
  } else {
    x <- x0
    z <- x0
  }
  w <- rep(0,n)
  obj <- rep(NA,MAXITER)
  error <- rep(NA,MAXITER)
  lagrng <- rep(NA,MAXITER)
  # pre-defined functions
  InfNorm <- function(x) max(abs(x))
  Lq <- function(x) sum(abs(x)^q)
  Lagrangian <- function(x,z,w) Lq(z) + t(w) %*% (x - z) + beta/2 * sum(abs(x - z)^2)
  HalfThres <- function(x, beta){
    tmp <- 3/2*(beta)^(-2/3)
    if (abs(x) > tmp)
      return (2/3*x*(1+cos(2/3*pi - 2/3 * acos(1/(4*beta)*(abs(x)/3)^(-3/2)))))
    else
      return (0)
  } 
  Threshold <- function(x, beta) {
    #  solve argmin_z |z|_q^q + beta/2 | z - x |^2
    if (q == 1) {
      return ((x - 1/beta)*(x > 1/beta) + (x + 1/beta)*(x < -1/beta))
    } else if (q == 1/2) {
      return(sapply(x,function(x) HalfThres(x,beta)))
    } else {
      # use Newton method for general q
      x.sign <- sign(x)
      x <- abs(x)
      z <- x
      derivative <- eval(parse(text = 'function(z) q*(q - 1)*z^(q - 2) + beta'))
      value <- eval(parse(text = 'function(z) q*z^(q - 1) + beta*(z - x) '))
      while (z!= 0 && abs(value(z)) > EPS) {
        z <- max(z - value(z)/derivative(z),0)
      }
      if (z != 0 && Lq(z) + beta/2*(z - x)^2 < beta/2*x^2) 
        return (x.sign * z)
      else
        return (0)
    }
  }
  if (VERBOSE) {
    cat(sprintf('%4s\t%10s\t%10s\t%10s\n', 'iter', 'obj','lagrng', 'error'))
  }
  # Calculate the projection x -> Px + q
  AAt <- A %*% t(A)
  P <- diag(rep(1,n)) - t(A) %*% solve(AAt, A)
  r <- t(A) %*% solve(AAt,y)
  for (k in 1:MAXITER) {
    # x update
    #------------------------------------------------
    #   minimize   \| x - z + w/beta \|_2^2
    #   subject to Ax = y
    #------------------------------------------------
    x <- P %*% (z - w/beta) + r
    # z update
    #------------------------------------------------
    #   minimize_z \|z\|_q^q + beta/2 \| x - z + w/beta \|_2^2
    #------------------------------------------------
    z <- apply(x + w/beta, 1, function(z) Threshold(z,beta))
    # w update
    w <- w + beta * (x - z)
    # record the values
    obj[k] <- Lq(x)
    error[k] <- InfNorm(x - z)
    lagrng[k] <- Lagrangian(x, z, w)
    if (VERBOSE) {
      cat(sprintf('%4d\t%10.4f\t%10.4f\t%10.4f\n', k, obj[k], lagrng[k], error[k]))
    }
    # beta update
    if (AUTO && k > 1 && lagrng[k] > lagrng[k - 1])
      increase.counts <- increase.counts + 1
    if (AUTO && increase.counts > MAXCOUNTS) {
      increase.counts <- -1
      beta <- beta * SCALE
    }
    # stopping criteria
    if (k == MAXITER) {
      FLAG <- 'Max iteration'
      break
    }
    if (AUTO && beta > LARGE) {
      FLAG <- 'beta too large'
      break
    }
    if (InfNorm(x) > LARGE) {
      FLAG <- 'Unbounded'
      break
    }
    if (error[k] < ABSTOL) {
      FLAG <- 'Absolute error'
      break
    }
    if (error[k] < RELTOL * InfNorm(c(x,z))) {
      FLAG <- 'Relative error'
      break
    }
  }
  timing <- proc.time() - ptm
  if (VERBOSE) {
    cat(sprintf('ADMM has stopped at iter %4d because of %10s.\n',k,FLAG))
    cat(sprintf('Elapsed time is %8.1e seconds .\n',timing[1]))
  }
  return (list(x = x, 
              FLAG = FLAG, 
              iter = k, 
              beta = beta, 
              obj = obj, 
              error = error, 
              Lagrangian = lagrng, 
              timing = timing))
}
