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
(1) SKK: "A two sample test in high dimensional data", Srivastava et.al. 2013
(2) Conformal: "A two-sample conditional distribution test using conformal prediction and weighted rank sum", Hu et.al. 2024
(3) C2ST: "Revisiting Classifier Two-Sample Tests", Lopez-Paz et.al. 2016
(4) KMMD: "A Kernel Two-sample Test(Kernel MMD)" Gretton et.al. 2012
(5) Miles: "A more powerful two-sample test in high dimensions using random projection" ME.Lopes et.al. 2011
(6) Xu16: "An adaptive two-sample test for high-dimensional means" Xu et.al. 2016
(7) Chen10: "A two-sample test for high-dimensional data with applications to gene-set testing", Chen et.al. 2010
(8) Chen14: "Two-Sample Tests for High Dimensional Means with Thresholding and Data Transformation", Chen et.al. 2014
(9) Sri08: "A test for the mean vector with fewer observations than the dimension.", Srivastava et.al. 2008
(10) Pan14: "A powerful and adaptive association test for rare variants", Pan et.al. 2014
(11) AutoTST: "AutoML Two-Sample Test", Kübler et.al.
(12) Bai96: ""Effect of high dimension: by an example of a two sample problem.", Bai et.al. 1996
(13) zwl: "Result consistency of high dimensional two-sample tests applied to gene ontology terms with gene sets" Zhang et.al.


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

### Causal Attribution: Comparing the feature-specific p-values versus the "Explaining Concept Shift with Interpretable Feature Attribution".
1. https://colab.research.google.com/drive/15K7Y0g0ic4LlUon5enWHcEZPdd6vcKwR#scrollTo=VV46gNnXAuy_
2. Lora/SFT for the GPT & Bert Text Embedding

