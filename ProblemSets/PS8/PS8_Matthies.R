library(nloptr)

#lines 5-35 using Claude Sonnet

N <- 100000  # Number of rows
K <- 10      # Number of columns

# Set seed for reproducibility
set.seed(100)

# Create matrix X
X <- matrix(rnorm((N * (K-1))), nrow = N, ncol = K-1)  # Generate random normal values
X <- cbind(rep(1, N), X)  # Add column of 1's as the first column

# Verify dimensions
dim(X)

# View first few rows to confirm structure
head(X)


sigma <- 0.5  # Standard deviation
sigma_squared <- 0.25  # Variance

# Set seed for reproducibility
set.seed(456)

# Generate random error vector eps ~ N(0, sigma^2)
eps <- rnorm(N, mean = 0, sd = sigma)

mean(eps)  # Should be close to 0
var(eps)   # Should be close to 0.25

# View first few elements
head(eps)


beta<- c(1.5, -1, -.25, .75, 3.5, -2, .5, 1, 1.25, 2)
Y <- (X%*%beta)+eps
head(Y)
#checking dimensions
dim(Y)

#OLS using closed form solution
XTranspose <- t(X)
solve(XTranspose%*%X)%*%XTranspose%*%Y #very close to actual betas, but still some error


#OLS using gradient descent

# set up a stepsize
alpha <- 0.0000003
# set up a number of iteration
iter <- 100
# define the gradient of f(x) = sum(y-X*beta)^2
gradient <- function(x){ return( -2*t(X) %*%(Y - (X%*%beta)))
}   #gradient of the OLS objective w/ repsect to beta

# randomly initialize a value to x
set.seed(100)
x <- floor(runif(1)*10)
# create a vector to contain all xs for all steps
x.All <- vector("numeric",iter)
# gradient descent method to find the minimum
for(i in 1:iter){
  x <- x - alpha*gradient(x)
  x.All[i] <- x
  print(x)
}

print(paste("The minimum of f(x) is ", x, sep = ""))
#values are converging to 3, perhaps I am defining the wrong objective function and
#gradient functions. Maybe this is right, but it arbitrarily does not seem correct to me.
#I am mainly skeptical I have not defined "function(x)" correctly in the gradient
#function


#OLS using L-BFGS

objfun <- function(beta,Y,X) {
  return (sum((Y-X%*%beta)^2))
  # equivalently, if we want to use matrix algebra:
  # return ( crossprod(y-X%*%beta) )
}
# Gradient of our objective function
gradient <- function(beta,Y,X) {
  return ( as.vector(-2*t(X)%*%(Y-X%*%beta)) )
}
# initial values
beta0 <- runif(dim(X)[2]) #start at uniform random numbers equal to number of coefficients
# Algorithm parameters
options <- list("algorithm"="NLOPT_LD_LBFGS","xtol_rel"=1.0e-6,"maxeval"=1e3)
# Optimize!
result <- nloptr( x0=beta0,eval_f=objfun,eval_grad_f=gradient,opts=options,Y=Y,X=X)
print(result)


#Nelder-Mead

# Our objective function
objfun <- function(x) {
  return(-2*t(X) %*%(Y - (X%*%beta)))
}
# initial values
xstart <- 0
# Algorithm parameters
options <- list("algorithm"="NLOPT_LN_NELDERMEAD","xtol_rel"=1.0e-8)
# Find the optimum!
res <- nloptr( x0=xstart,eval_f=objfun,opts=options)
print(res) #saying there is an error for (Y-(X*Beta)) not being conformable arrays,
#but all the matrices should work while using this order of operations


#MLE

# Our objective function
objfun  <- function(theta,Y,X) {
  # need to slice our parameter vector into beta and sigma components
  beta2    <- theta[1:(length(theta)-1)]
  sig     <- theta[length(theta)]
  # write objective function as *negative* log likelihood (since NLOPT minimizes)
  loglike <- -sum( -.5*(log(2*pi*(sig^2)) + ((Y-X%*%beta2)/sig)^2) ) 
  return (loglike)
}

# initial values
theta0 <- runif(dim(X)[2]+1) #start at uniform random numbers equal to number of coefficients
theta0 <- append(as.vector(summary(lm(Y ~ X -1))$coefficients[,1]),runif(1))
# Algorithm parameters
options <- list("algorithm"="NLOPT_LN_NELDERMEAD","xtol_rel"=1.0e-6,"maxeval"=1e4)
# Optimize!
result <- nloptr( x0=theta0,eval_f=objfun,opts=options,Y=Y,X=X)
print(result)
betahat  <- result$solution[1:(length(result$solution)-1)]
sigmahat <- result$solution[length(result$solution)]



#OLS using lm
lmOLS <- lm(Y ~ X -1)
table<-summary(lmOLS)
print(xtable(table), type = "latex")
tidy(lmOLS) #viewing intercepts in console


