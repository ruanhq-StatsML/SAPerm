# Hypothesis Testing Procedure for Distribution Shift with Application on Sentiment Analysis

Implementation for the Permutation Test for Distribution Shift with Application to Sentiment Analysis: Both Tree-Based Permutation Test and R-risk Causal Inference Test. We benchmark the following 15 methods with the 6 developed methods:
- RFPerm: Random Forest Permutation Test for Distribution Shift
- XGBPerm: XGBoost based Permutation Test for Distribution Shift
~  Causal Inference Based Test followed by a Permute-then-Refit Procedure on the Variable Importance
- CFPerm_loco:      Leave-one-Covariate-out Variable Importance
- CFPerm_PermuCATE: Conditional Permutation Variable Importance
- CFPerm_grf:       Causal Forest Split Variable Importance
~ DRPerm:           Permutation Test via the doubly robust pseudo-outcome learner
~ RRPerm:           Permutation Test via the R-learner
--------
Benchmark Methods:
(1) SKK
(2) Conformal
(3) C2ST
(4) S

## Python version:
```python

```

## R version:
#### Installation (local)

```r
install.packages(c("devtools", "roxygen2", "testthat", "grf", "MASS", ""))
devtools::install_local("path/to/CFPerm")
```

## Testing Example

```r
library(CFPerm)

```

## Development

```r
devtools::document()
devtools::test()
devtools::check()
```
