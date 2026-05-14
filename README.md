# CFPerm

Implementation for the Permutation Test for Distribution Shift \& Fairness Violation with Application to Sentiment Analysis


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
