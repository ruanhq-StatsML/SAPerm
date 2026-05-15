# Hypothesis Testing Procedure for Distribution Shift with Application on Sentiment Analysis

Implementation for the Permutation Test for Distribution Shift with Application to Sentiment Analysis: Both Tree-Based Permutation Test and R-risk Causal Inference Test.


## Python version:
```python
!pip install cfperm_vimp
from cfperm_vimp import RRPerm, DRPerm

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
