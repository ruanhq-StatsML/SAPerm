#' Two-sample datasets generator
#'
#' This function generates simulated high dimensional two-sample data from user specified populations with given mean vectors, covariance structure, sample sizes, and dimension of each observation. It could generate the long-range dependent process proposed by Hall et al. (1998) in additional to some processes provided in arima.sim().
#'
#' @param n number of observations in the 1st sample.
#' @param m number of observations in the 2nd sample.
#' @param p the dimensionality of the each observation. The samples from both populations should have the same dimension.
#' @param muX \code{p} by 1 vector of component means for the 1st population.
#' @param muY \code{p} by 1 vector of component means for the 2nd population.
#' @param dep dependence structure among the \code{p} components for both populations.
#'            Possible options are:
#'
#'            'IND' for independence;
#'
#'            'SD' for strong dependency, AR(1) with parameter 0.9;
#'
#'            'WD' for weak dependency, ARMA(2, 2) with AR parameters 0.4 and -0.1, and MA parameters 0.2 and 0.3;
#'
#'            'LR' for long-range dependency with parameter 0.7.
#'
#'            For more details about the configurations,  please refer to Zhang and Wang (2020).
#'
#' @param commoncov a logical indicating whether the two populations have equal covariance matrices. If FALSE, the innovations used in generating data for the 2nd population will be scaled by the square root of the value specified in VarScaleY.
#' @param VarScaleY constant by which innovations are scaled in generating observations for the 2nd sample when commoncov=FALSE.
#' @param S the number of data sets to simulate.
#' @param innov a function used to generate the innovations, such as \code{innov=function(n,...) rnorm(n,0,1)}.
#' @param heteroscedastic a logical indicating whether the components will be scaled by the entries in the diagonal matrix specified by \code{het.diag}.
#' @param het.diag a \code{p} by \code{p} diagonal matrix, where the entries on the diagonal will be used to scale the component standard deviations. Only valid when \code{heteroscedastic = TRUE}.
#'
#' @return A list of \code{S} lists, each consisting of an \code{n} by \code{p} matrix \code{X}, an \code{m} by \code{p} matrix \code{Y}, the sample sizes, \code{n} and \code{m}, for each population, and the dimensionality \code{p}.
#'
#' @export
#'
#' @references
#' Hall, P., Jing, B.-Y., and Lahiri, S. N. (1998). On the sampling window method for long-range dependent data. Statistica Sinica, 8(4):1189-1204.
#'
#' @examples
#' # Generate 3 two-sample datasets of dimensionality 300
#' # with sample sizes 45 for one sample & 60 for the other.
#' buildData(n = 45, m =60, p = 300,
#'           muX = rep(0,300), muY = rep(0,300),
#'           dep = 'IND', S = 3, innov = rnorm)
buildData <- function(n,
                      m,
                      p,
                      muX,
                      muY,
                      dep,
                      commoncov = TRUE,
                      VarScaleY = 1,
                      S = 1,
                      innov = function(n, ...) stats::rnorm(n, 0, 1),
                      heteroscedastic = FALSE,
                      het.diag)
{
   ERRORS <- list()
   if (dep == 'IND') {
      for (s in 1:S)
      {
         Zx <- matrix(innovate(n * p, rand.gen = innov), nrow = n)
         Zy <- matrix(innovate(m * p, rand.gen = innov), nrow = m)
         ERRORS[[length(ERRORS) + 1]] <- list(Zx = Zx , Zy = Zy)
      }
   } else if (dep == 'SD') {
      Zx <- matrix(0, n, p)
      Zy <- matrix(0, m, p)
      for (s in 1:S)
      {
         for (i in 1:max(n, m))
         {
            if (i <= n)
               Zx[i, ] <-
                  stats::arima.sim(
                     model = list(ma = c(0, 0) , ar = c(.9, 0)) ,
                     n = p ,
                     rand.gen = innov
                  )
            if (i <= m)
               Zy[i, ] <-
                  stats::arima.sim(
                     model = list(ma = c(0, 0) , ar = c(.9, 0)),
                     n = p ,
                     rand.gen = innov
                  )
         }
         ERRORS[[length(ERRORS) + 1]] <-
            list(Zx = Zx , Zy = Zy)
      }
   } else if (dep == 'WD') {
      Zx <- matrix(0, n, p)
      Zy <- matrix(0, m, p)
      for (s in 1:S)
      {
         for (i in 1:max(n, m))
         {
            if (i <= n)
               Zx[i, ] <-
                  stats::arima.sim(
                     model = list(ma = c(0.2, 0.3) , ar = c(.4,-.1)),
                     n = p ,
                     rand.gen = innov
                  )
            if (i <= m)
               Zy[i, ] <-
                  stats::arima.sim(
                     model = list(ma = c(0.2, 0.3) , ar = c(.4,-.1)),
                     n = p ,
                     rand.gen = innov
                  )
         }
         ERRORS[[length(ERRORS) + 1]] <-
            list(Zx = Zx , Zy = Zy)
      }
   } else if (dep == 'LR')  {
      H <- 0.7
      R <- matrix(0, p, p)
      for (i in 1:p)
      {
         for (j in 1:p)
         {
            k <- abs(i - j)
            R[i, j] <-
               .5 * ((k + 1) ^ (2 * H) + ((k - 1) ^ (2)) ^ H - 2 * k ^ (2 * H))
         }
      }
      diag(R) <- 1
      U <- chol(R)
      for (s in 1:S)
      {
         Z0x <- innovate(n * p, rand.gen = innov)
         Zx <- matrix(Z0x, nrow = n) %*% U
         Z0y <- innovate(m * p, rand.gen = innov)
         Zy <- matrix(Z0y , nrow = m) %*% U
         ERRORS[[length(ERRORS) + 1]] <-
            list(Zx = Zx , Zy = Zy)
      }
   }
   if (heteroscedastic) {
      for (s in 1:S)
      {
         ERRORS[[s]]$Zx <- ERRORS[[s]]$Zx  %*% het.diag
         ERRORS[[s]]$Zy <- ERRORS[[s]]$Zy  %*% het.diag
      }
   }
   if (!commoncov) {
      for (s in 1:S)
      {
         ERRORS[[s]]$Zy <- sqrt(VarScaleY) * ERRORS[[s]]$Zy
      }
 
   }
   MUX <- matrix(muX,
                 nrow = n,
                 ncol = p,
                 byrow = T)
   MUY.list <- list()
   for (s in 1:S)
   {
      MUY.list[[length(MUY.list) + 1]] <-
         matrix(muY,
                nrow = m,
                ncol = p,
                byrow = T)
   }
   DATA <- list()
   for (s in 1:S)
   {
      X <- MUX + ERRORS[[s]]$Zx
      Y <- MUY.list[[s]] + ERRORS[[s]]$Zy
      DATA[[length(DATA) + 1]] <- list(
         X = X,
         Y = Y,
         n = n,
         m = m,
         p = p
      )
   }
   return(DATA)
}
 
innovate <- function (n, rand.gen = stats::rnorm, innov = rand.gen(n, ...), ...)
{
   return(innov)
}
 
#' Random sample from shifted gamma distribution
#'
#' This function generates random samples from shifted gamma distribution. That is, random samples are first generated from gamma distribution with shape parameter \code{shape} and scale parameter \code{scale} and then the mean of the gamma distribution, \code{shape}*\code{scale}, is subtracted from the sample.
#'
#' @param n number of observations.
#' @param shape the shape parameter of gamma distribution
#' @param scale the scale parameter of gamma distribution
#' #'
#' @return A vector of \code{n} values. It is equivalent to rgamma(n, shape, scale)- shape * scale.
#'
#' @export
#'
#' @examples
#' # Generate a sample of shifted gamma observations with shape parameter 4 and scale parameter 2.
#' set.seed(10)
#' rgammashift(n = 5, shape =4, scale = 2)
#' # It is equivalent to
#' set.seed(10)
#' rgamma(n = 5, shape=4, scale=2)- 4 * 2
rgammashift <- function (n, shape, scale)
{
   x <- stats::rgamma(n = n, shape = shape, scale = scale) - shape * scale
   return(x)
}
 
 
#' High-dimensional two-sample test proposed by Zhang and Wang (2020)
#'
#' This function implements the test of equal mean for two-sample high-dimension data using the ZWL and ZWLm tests proposed by Zhang and Wang (2020).
#'
#' @param X The data matrix (n by p) from the first population.
#' @param Y The data matrix (m by p) from the second population.
#' @param order The order of center correction. Possible choices are 0,  2.
#'              To use the ZWLm test, set \code{order = 0}; to use the ZWL test, set \code{order = 2}. For moderate sample sizes, ZWLm is recommended.
#'
#' @return
#' \describe{
#' \item{statistic}{The value of the test statistic.}
#' \item{pvalue}{The p-value of the test statistic based on the asymptotic normality established by Zhang and Wang (2020)}
#' \item{Tn}{The average of the squared univariate t-statistics.}
#' \item{var}{The estimated variance of Tn}
#' }
#'
#'
#' @export
#'
#' @references
#' Zhang, H. and Wang, H. (2020). Result consistency of high dimensional
#' two-sample tests applied to gene ontology terms with gene sets. Manuscript in review.
#'
#' @examples
#' # Generate a simulated two-sample dataset and apply the ZWL test
#' data <- buildData(n = 45, m =60, p = 300,
#'           muX = rep(0,300), muY = rep(0,300),
#'           dep = 'IND', S = 1, innov = rnorm)
#' zwl_test(data[[1]]$X, data[[1]]$Y, order = 2)
#'
#' # Apply the ZWLm test to a GO term to see if the two groups are differentiately expressed.
#' # The data for the GO term were stored in GO_example.
#' zwl_test(GO_example$X, GO_example$Y, order = 0)
#' # Apply the ZWL test to the GO term
#' zwl_test(GO_example$X, GO_example$Y, order = 2)
#'
#'
zwl_test <- function(X, Y, order = 0)
{
   p <- ncol(X)
   n <- nrow(X)
   m <- nrow(Y)
   mean1 <- matrix(rep(1, n), nrow = 1) %*% X / n
   mean2 <- matrix(rep(1, m), nrow = 1) %*% Y / m
   var1 <- matrix(rep(1, n), nrow = 1) %*% scale(X, mean1, F) ^ 2 / (n - 1)
   var2 <- matrix(rep(1, m), nrow = 1) %*% scale(Y, mean2, F) ^ 2 / (m - 1)
   Sp <- sqrt(var1 / n + var2 / m)
   t_sq <- ((mean1 - mean2) / Sp) ^ 2
   Tn <-  drop(rep(1, p) %*% t(t_sq) / p)
   var_hat <-  drop(rep(1, p) %*% cov_hat(X, Y) %*%
                     matrix(rep(1, p), ncol = 1) / p ^ 2)
   center.est <-
      apply(
         rbind(X, Y),
         2,
         compute_center,
         n = nrow(X),
         m = nrow(Y),
         ntoorderminus = order
      )
   test_stat <- (Tn - drop(rep(1, p) %*% center.est) / p) / sqrt(var_hat)
   pvalue <-  2 * (1 - stats::pnorm(abs(test_stat), 0, 1))
   list(
      statistic = test_stat,
      pvalue = pvalue,
      Tn = Tn,
      var = var_hat
   )
}
 
 
compute_center <- function(xy, n, m, ntoorderminus = 2)
{
   x <- xy[1:n]
   y <- xy[(n + 1):(n + m)]
   mean_x = drop(rep(1, n) %*% x / n)
   mean_y = drop(rep(1, m) %*% y / m)
   x_minus = x - mean_x
   y_minus = y - mean_y
   sig.sq.x.hat <- rep(1, n) %*% x ^ 2 / n - mean_x ^ 2
   sig.sq.y.hat <- rep(1, m) %*% y ^ 2 / m  - mean_y ^ 2
   tau.sq.hat <- sig.sq.x.hat + (n / m) * sig.sq.y.hat
   mu.3.hat  <- rep(1, n) %*% x_minus ^ 3 / n
   eta.3.hat <- rep(1, m) %*% y_minus ^ 3 / m
   mu.4.hat <-  rep(1, n) %*% x_minus ^ 4 / n
   eta.4.hat <- rep(1, m) %*% y_minus ^ 4 / m
   mu.5.hat <-  rep(1, n) %*% x_minus ^ 5 / n
   eta.5.hat <- rep(1, m) %*% y_minus ^ 5 / m
   if (ntoorderminus == 0) {
      return(1)
   } else if (ntoorderminus == 1) {
      a1 <- tau.sq.hat ^ (-1) * (sig.sq.x.hat + (n / m) ^ 2 * sig.sq.y.hat)
      a2 <- tau.sq.hat ^ (-3) * 2 * (mu.3.hat  + (n / m) ^ 2 * eta.3.hat) ^ 2
      c <- 1 +  n ^ (-1) * (a1 + a2)
      drop(c)
   } else if (ntoorderminus == 2) {
      a1 <- tau.sq.hat ^ (-1) * (sig.sq.x.hat + (n / m) ^ 2 * sig.sq.y.hat)
      a2 <- tau.sq.hat ^ (-3) * 2 * (mu.3.hat  + (n / m) ^ 2 * eta.3.hat) ^ 2
      b1 <- tau.sq.hat ^ (-2) * ((sig.sq.x.hat + (n / m) ^ 2 * sig.sq.y.hat) -
                                    ((mu.4.hat - 3 * sig.sq.x.hat ^ 2) +
                                          (n / m) ^ 4 * (eta.4.hat - 3 * sig.sq.y.hat ^ 2)))
      b2 <- tau.sq.hat ^ (-3) * ((sig.sq.x.hat + (n / m) ^ 2 * sig.sq.y.hat) *
                                    ((mu.4.hat -  sig.sq.x.hat ^ 2) + (n / m) ^ 3 * (eta.4.hat - sig.sq.y.hat ^ 2)) -
                                    4 * (mu.3.hat + (n / m) ^ 2 * eta.3.hat) * (mu.3.hat + (n / m) ^ 3 * eta.3.hat) -
                                    2 * (mu.3.hat ^ 2 + (n / m) ^ 5 * eta.3.hat ^ 2))
      b3 <- tau.sq.hat ^ (-4) * (6 * (sig.sq.x.hat + (n / m) ^ 2 * sig.sq.y.hat) *
                                    (mu.3.hat  + (n / m) ^ 2 * eta.3.hat) ^ 2 -
                                    6 * (mu.3.hat + (n / m) ^ 2 * eta.3.hat) *
                                    (mu.5.hat - 2 * mu.3.hat * sig.sq.x.hat +
                                     (n / m) ^ 4 * (eta.5.hat - 2 * eta.3.hat * sig.sq.y.hat)) -
                                    3 * ((mu.4.hat - sig.sq.x.hat ^ 2) + (n / m) ^ 3 * (eta.4.hat - sig.sq.y.hat ^ 2)) ^ 2)
      b4 <- tau.sq.hat ^ (-5) * (3 * (sig.sq.x.hat + (n / m) * sig.sq.y.hat) *
                                    ((mu.4.hat - sig.sq.x.hat ^ 2) + (n / m) ^ 3 * (eta.4.hat - sig.sq.y.hat ^ 2)) ^ 2 +
                                    12 * (mu.3.hat  + (n / m) ^ 2 * eta.3.hat) ^ 2  *
                                    ((mu.4.hat - sig.sq.x.hat ^ 2) + (n / m) ^ 3 * (eta.4.hat - sig.sq.y.hat ^ 2)))
      c <- 1 +  n ^ (-1) * (a1 + a2) + n ^ (-2) * (b1 + b2 + b3 + b4)
      drop(c)
   }
}
 
cov_hat <- function(X, Y)
{
   n <- nrow(X)
   m <- nrow(Y)
   p <- ncol(X)
   lambda1 <- n / (n + m)
   lambda2 <- m / (n + m)
   cov_X <- stats::cov(X, X)
   cov_Y <- stats::cov(Y, Y)
   numerator <-  (lambda2 * cov_X + lambda1 * cov_Y) ^ 2 * 2
   a <-  lambda2 * diag(cov_X)
   b <-  lambda1 * diag(cov_Y)
   denom <-  (a + b) %*% t(a + b)
   numerator / denom * outer(1:p, 1:p, function(x, y) abs(x - y) <= ceiling(p ^ (3 / 8)))
}
 
#' Apply the test by Zhang and Wang (2020) to multiple simulated two-sample datasets
#'
#' Apply the two-sample high-dimensional test by Zhang and Wang (2020) to multiple simulated two-sample high dimensional datasets. This function is useful for Monte Carlo experiments.
#'
#' @param DATA The list of dataset lists generated by \code{buildData}.
#' @param order The order of the center correction. Possible choices are 0, 2.
#'              To use the ZWLm test, set \code{order = 0}; to use the ZWL test, set \code{order = 2}. THE ZWLm test is recommended for moderate sample sizes.
#'
#' @return A dataframe with each row consisting the values of the
#' test statistics, p-values, Tn, and the estimate of Var(Tn).
#'
#' @export
#'
#' @references
#' Zhang, H. and Wang, H. (2020). Result consistency of high dimensional
#' two-sample tests applied to gene ontology terms with gene sets. Manuscript in review.
#'
#' @examples
#' # Generate 3 simulated two-sample datasets and apply the ZWL test
#' data <- buildData(n = 45, m =60, p = 300,
#'           muX = rep(0,300), muY = rep(0,300),
#'           dep = 'IND', S = 3, innov = rnorm)
#' zwl_sim(data, order = 2)
#'
#'
zwl_sim <- function(DATA, order = 0)
{
   res <- NULL
   for (i in 1:length(DATA))
   {
      res <- rbind(res, unlist(zwl_test(DATA[[i]]$X, DATA[[i]]$Y, order)))
   }
   data.frame(res)
}
 
 
 
#' High-dimensional two-sample test (SKK) proposed by Srivastava, Katayama, and Kano(2013)
#'
#' This function implements the two-sample high-dimensional test proposed by Srivastava, Katayama, and Kano(2013).
#'
#' @param X The data matrix (n by p) from the first population.
#' @param Y The data matrix (m by p) from the second population.
#'
#' @return A list consisting of the values of the test statistic and p-value.
#' @export
#'
#' @references
#' Srivastava, M. S., Katayama, S., and Kano, Y. (2013).
#' A two sample test in high dimensional data. Journal of Multivariate Analysis, 114:349-358.
#'
#' @examples
#' # Generate a simulated dataset and apply the SKK test
#' data <- buildData(n = 45, m =60, p = 300,
#'                  muX = rep(0,300), muY = rep(0,300),
#'                  dep = 'IND', S = 1, innov = rnorm)
#' SKK_test(data[[1]]$X, data[[1]]$Y)
#'
#' # Apply the SKK test to the data for a GO term stored in GO_example
#' SKK_test(GO_example$X, GO_example$Y)
SKK_test <-  function(X, Y)
{
   n1 <- nrow(X)
   n2 <- nrow(Y)
   p <- ncol(X)
   n <- n1 + n2 - 2
   X.bar <- apply(X, 2, mean)
   Y.bar <- apply(Y, 2, mean)
   S1 <- stats::cov(X)
   S2 <- stats::cov(Y)
   D1 <- diag(diag(S1))
   D2 <- diag(diag(S2))
   D <- D1 / n1 + D2 / n2
   R <- solve(sqrt(D)) %*% (S1 / n1 + S2 / n2) %*%  solve(sqrt(D))
   c.p.n <- 1 + sum(diag(R %*% R) / p ^ (3 / 2))
   var.qn <-
      2 * sum(diag(R %*% R)) / p - 2 * sum(diag(solve(D) %*% S1)) ^ 2 / p / n1 /
      (n1 + 1) ^ 2 - 2 * sum(diag(solve(D) %*% S2)) ^ 2 / p / n2 / (n2 + 1) ^ 2
   denom <- sqrt(p * c.p.n * var.qn)
   TSvalue <-
      (t(X.bar - Y.bar) %*% solve(D) %*% (X.bar - Y.bar) - p) / denom
   pvalue <- 2 * (1 - stats::pnorm(abs(TSvalue), 0, 1))
   Tn = t(X.bar - Y.bar) %*% solve(D) %*% (X.bar - Y.bar)/p
   list(TSvalue = TSvalue, pvalue = pvalue)
}
 
#' Apply the SKK test to multiple simulated two-sample datasets
#'
#' This function performs the SKK test of Srivastava, Katayama, and Kano(2013) on multiple high-dimensional two-sample datasets.
#' It is useful for Monte Carlo experiments.
#'
#' @param DATA The list of dataset lists generated by \code{buildData}.
#'
#' @return a dataframe, each row of which reports the values of the
#' SKK test statistics and the p-values.
#' @export
#'
#' @examples
#' # Generate 3 simulated datasets and apply the SKK test
#' data <- buildData(n = 45, m =60, p = 300,
#'                  muX = rep(0,300), muY = rep(0,300),
#'                  dep = 'IND', S = 3, innov = rnorm)
#' SKK_sim(data)
#'
#'
#' @references
#' Srivastava, M. S., Katayama, S., and Kano, Y. (2013).
#' A two sample test in high dimensional data. Journal of Multivariate Analysis, 114:349-358.
#'
SKK_sim <- function(DATA)
{
   res <- NULL
   for (i in 1:length(DATA))
   {
      res <- rbind(res, unlist(SKK_test(DATA[[i]]$X, DATA[[i]]$Y)))
   }
   data.frame(res)
}