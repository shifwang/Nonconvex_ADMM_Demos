# give the phase transition of Lq basis pursuit
create_y_A_x <- function(k, m, n, 
                           type = 'Gaussian',
                           seed = NULL){
  if (is.null(seed)){
    set.seed(proc.time()[3])
  } else {
    set.seed(seed)
  }
  if (type == 'binary'){
    x = c(rnorm(sparsity)>0,rep(0,dim - sparsity))[sample(dim,dim,replace = F)]
  } else if (type == 'Gaussian'){
    x = c(rnorm(sparsity)>0,rep(0,dim - sparsity))[sample(dim,dim,replace = F)]
  } else {
    stop('what is type?')
  }
  A = matrix(rnorm(dim*measurement),ncol = dim)
  y = A %*% as.matrix(x)
  return(list(x = x, y = y, A = A))
}
simulator <- function(q, k, y, A, x){
  #------------------------------
  # use Lq_basis_pursuit to solve
  #   minimize   ||x||_q^q
  #   subject to y = Ax
  #------------------------------
  est <- Lq_basis_pursuit(A,y,q)
  if (max(abs(est$x - x)) <= 1e-3*max(abs(x))){
    return(TRUE)
  } else {
    return(FALSE)
  }
}
EstimateRecoverRate <- function(q, n, m, k, type){
  # Estimate recovery rate of Lq_basis_pursuit at point (n,m,k) with signal type
  Niter <- length(seeds)
  total.success <- 0
  for(i in 1:Niter){
    tmp <- create_y_A_x(k,y,A,x,type,seeds[i])
    success <- simulator(q,k,y,A,x)
    total.success <- total.success + success
  }
  rate <- total.success/Niter
  return(rate)
}
percentileK <- function(q, n, m, type, percent){
  # Use bisection to find K correspond to certain recovery rate
  Kmin <- 1
  Kmax <- m - 1
  if (EstimateRecoverRate(q,n,m,Kmax,type) >= percent){
    K = Kmax
    return(K)
  }
  if (EstimateRecoverRate(q,n,m,Kmin,type) < percent){
    K = 0
    return(K)
  }
  while ((Kmax - Kmin)/m > 0.01 && (Kmax - Kmin) >= 3){
    rev.rate <- EstimateRecoverRate(q,n,m,ceiling((Kmin + Kmax)/2),type)
    if (rev.rate > percent){
      Kmin <- ceiling((Kmin + Kmax)/2)
    } else {
      Kmax <- ceiling((Kmin + Kmax)/2)
    }
  }
  K = Kmin
  return(K)
}