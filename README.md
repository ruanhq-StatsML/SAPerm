# Hypothesis Testing Procedure for Distribution Shift with Application on Sentiment Analysis

Implementation for the Permutation Test for Distribution Shift with Application to Sentiment Analysis: Both Tree-Based Permutation Test and R-risk Causal Inference Test. We benchmark the following 15 methods with the 6 developed methods: 
- RFPerm: Random Forest Permutation Test for Distribution Shift 
- XGBPerm: XGBoost based Permutation Test for Distribution Shift \
---- Causal Inference Based Test followed by a Permute-then-Refit Procedure on the Variable Importance 
- CFPerm_loco:      Leave-one-Covariate-out Variable Importance 
- CFPerm_PermuCATE: Conditional Permutation Variable Importance 
- CFPerm_grf:       Causal Forest Split Variable Importance 
~ DRPerm:           Permutation Test via the doubly robust pseudo-outcome learner 
~ RRPerm:           Permutation Test via the R-learner 
-------- 
Benchmark Methods
We compare against the following state‑of‑the‑art and classical two‑sample testing methods via Python

SKK (2013) – A two‑sample test in high‑dimensional data
Srivastava, M. S., Katayama, S., & Kano, Y. (2013).
Journal of Multivariate Analysis. [A test for high‑dimensional mean vectors with unequal covariance matrices.]

Conformal (2024) – A two‑sample conditional distribution test using conformal prediction and weighted rank sum
Hu, X., et al. (2024).
arXiv preprint. [Conditional distribution test via conformal prediction and weighted rank sum.]

C2ST (2016) – Revisiting Classifier Two‑Sample Tests
Lopez‑Paz, D., & Oquab, M. (2016).
ICLR 2017. [Uses a classifier’s accuracy to test whether two samples come from the same distribution.]

KMMD (2012) – A Kernel Two‑Sample Test (Kernel MMD)
Gretton, A., Borgwardt, K. M., Rasch, M. J., Schölkopf, B., & Smola, A. (2012).
Journal of Machine Learning Research. [Maximum Mean Discrepancy with a characteristic kernel.]

Miles (2011) – A more powerful two‑sample test in high dimensions using random projection
Lopes, M. E., Jacob, L., & Wainwright, M. J. (2011).
Advances in Neural Information Processing Systems. [Random projection based test for high‑dimensional means.]

Xu16 (2016) – An adaptive two‑sample test for high‑dimensional means
Xu, G., Lin, L., Wei, P., & Pan, W. (2016).
Biometrika. [Adaptive thresholding for sparse high‑dimensional mean differences.]

Chen10 (2010) – A two‑sample test for high‑dimensional data with applications to gene‑set testing
Chen, S. X., & Qin, Y. L. (2010).
The Annals of Statistics. [Uses a regularized Hotelling’s T² for gene‑set analysis.]

Chen14 (2014) – Two‑Sample Tests for High Dimensional Means with Thresholding and Data Transformation
Chen, S. X., Li, J., & Zhong, P. S. (2014).
Journal of the American Statistical Association. [Combines thresholding and transformation to handle sparse signals.]

Sri08 (2008) – A test for the mean vector with fewer observations than the dimension
Srivastava, M. S. (2008).
Journal of Multivariate Analysis. [Tests for the mean vector when p > n.]

Pan14 (2014) – A powerful and adaptive association test for rare variants
Pan, W. (2014).
Genetic Epidemiology. [Originally for rare variant association, can be adapted for high‑dimensional two‑sample means.]

AutoTST (2020) – AutoML Two‑Sample Test
Kübler, J., et al. (2022).
[An automated machine learning approach to two‑sample testing. Please check the original source for exact year and citation.]

Bai96 (1996) – Effect of high dimension: by an example of a two sample problem
Bai, Z., & Saranadasa, H. (1996).
Statistica Sinica. [Demonstrates failure of classical Hotelling’s T² when dimension grows with sample size.]

ZWL (Zhang et al.) – Result consistency of high‑dimensional two‑sample tests applied to gene ontology terms with gene sets
Zhang, W., et al. (2020).
[A study on consistency of high‑dimensional tests when applied to gene ontology terms. Please provide the full citation if available.]


### Illustration for Fine-Tuning & Word Embedding and Testing Pipeline:
 [https://colab.research.google.com/drive/15K7Y0g0ic4LlUon5enWHcEZPdd6vcKwR#scrollTo=VV46gNnXAuy_](https://colab.research.google.com/drive/1y-Hl654ASB48EmimkRdQ79XIKbEHWzSq#scrollTo=sWb7WEPD3w05)

#### For the details of the dataset, please contact hank_rhq@outlook.com for more details
