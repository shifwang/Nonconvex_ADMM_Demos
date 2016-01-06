# An example of Lq_basis_pursuit
source('Algorithm.R')
dim <-200
sparsity <- 35
measurement <- 100
seed <- 10140
set.seed(seed)
score.board <- matrix(0,nrow = 2,ncol = 2)
distribution = 'binary'
for (test.ind in 1:100){
  cat(test.ind,'\n')
  if (distribution == 'binary'){
    x = c(sign(rnorm(sparsity)),rep(0,dim - sparsity))[sample(dim,dim,replace = F)]
  } else if (distribution == 'Gaussian'){
    x = c(rnorm(sparsity),rep(0,dim - sparsity))[sample(dim,dim,replace = F)]
  } else {
    stop('what is distribution?')
  }
  A = matrix(rnorm(dim*measurement),ncol = dim)
  y = A %*% as.matrix(x) 
  est <- LqBasisPursuit(A,y,1, x0 = NULL, beta = 10, AUTO = T, MAXCOUNTS = 100, MAXITER = 1000, VERBOSE = F)
  if (max(abs(est$x - x)) < 0.01){
    one <- T
  } else {
    one <- F
  }
  est <- LqBasisPursuit(A,y,.5,x0 = est$x, beta = 100, AUTO = T, MAXCOUNTS = 100, MAXITER = 1000, VERBOSE = F)
  if (max(abs(est$x - x)) < 0.01){
    half <- T
  } else {
    half <- F
  }
  if (half) {
    if (one)
      score.board[1,1] = score.board[1,1] + 1
    else 
      score.board[1,2] = score.board[1,2] + 1
  } else {
    if (one)
      score.board[2,1] = score.board[2,1] + 1
    else 
      score.board[2,2] = score.board[2,2] + 1
  }
  #plot(est$x, col = 'black')
  #points(x,col = 'red')
  #plot(log10(est$error),type = 'l',xlab = 'iter', ylab = 'log10(error)')
}
print(score.board)
