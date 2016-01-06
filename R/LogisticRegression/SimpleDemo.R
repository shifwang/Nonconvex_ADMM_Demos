# An example of LqBasisPursuit
# Load functions
require(ggplot2)
source('Algorithm.R')
# Generate data
seed <- 100
set.seed(seed)
dim <-200
sparsity <- 40
measurement <- 100
score.board <- matrix(0,nrow = 2,ncol = 2)
distribution = 'binary'
if (distribution == 'binary'){
  x = c(sign(rnorm(sparsity)),rep(0,dim - sparsity))[sample(dim,dim,replace = F)]
} else if (distribution == 'Gaussian'){
  x = c(rnorm(sparsity),rep(0,dim - sparsity))[sample(dim,dim,replace = F)]
} 
A = matrix(rnorm(dim*measurement),ncol = dim)
y = A %*% as.matrix(x) 
# Run code
est.one <- LqBasisPursuit(A,y,1, x0 = NULL, beta = 10, AUTO = T, MAXCOUNTS = 100, MAXITER = 1000, VERBOSE = F)
est.half <- LqBasisPursuit(A,y,.5,x0 = NULL, beta = 100, AUTO = T, MAXCOUNTS = 100, MAXITER = 1000, VERBOSE = F)
# Plot result
print(ggplot() + 
        geom_point(data = data.frame(x = 1:dim, y = as.matrix(x)), aes(x = x, y = y, colour = 'original')) + 
        geom_point(data = data.frame(x = 1:dim, y = est.one$x), aes(x = x, y = y, colour = 'basis pursuit')) + 
        theme(legend.title=element_blank()) +
        scale_color_manual(values = c('original' = "black", 'basis pursuit' = "red")) 
)
print(
  ggplot() +
    geom_line(data = data.frame(iter = 1:est.one$iter, error = log10(est.one$error[1:est.one$iter])), aes(x = iter, y = error)) + 
    xlab('iter') + ylab(expression(paste('log'[10],' ( ||',x^k,' - ', z^k,' ||', infinity, ' )'))) +
    ggtitle(expression(paste(L[1], ' error')))
)
print(
  ggplot() +
    geom_line(data = data.frame(iter = 1:est.one$iter, error = est.one$obj[1:est.one$iter]), aes(x = iter, y = error)) + 
    xlab('iter') + ylab(expression(paste(' || x ||'[1]))) + 
    ggtitle(expression(paste(L[1], ' objective')))
)
print(ggplot() + 
        geom_point(data = data.frame(x = 1:dim, y = as.matrix(x)), aes(x = x, y = y, colour = 'original')) + 
        geom_point(data = data.frame(x = 1:dim, y = est.half$x), aes(x = x, y = y, colour = 'basis pursuit')) + 
        theme(legend.title=element_blank()) +
        scale_color_manual(values = c('original' = "black", 'basis pursuit' = "red"))
)
print(
  ggplot() +
    geom_line(data = data.frame(iter = 1:est.half$iter, error = log10(est.half$error[1:est.half$iter])), aes(x = iter, y = error)) + 
    xlab('iter') + ylab(expression(paste('log'[10],' ( ||',x^k,' - ', z^k,' ||', infinity, ' )'))) + 
    ggtitle(expression(paste(L[0.5], ' error')))
)
print(
  ggplot() +
    geom_line(data = data.frame(iter = 1:est.half$iter, error = est.half$obj[1:est.half$iter]), aes(x = iter, y = error)) + 
    xlab('iter') + ylab(expression(paste(' || x ||'[0.5]^0.5))) +
    ggtitle(expression(paste(L[0.5], ' objective')))
)

