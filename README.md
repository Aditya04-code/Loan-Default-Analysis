# Predicting Loan Default Using Machine Learning Algorithms

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![UCD Smurfit](https://img.shields.io/badge/Module-FIN42100-orange.svg)

## Project Overview

This project was conducted as part of the **FIN42100 Machine Learning in Finance** module at **UCD Michael Smurfit Graduate Business School**. The group assignment, titled **Predicting Loan Default Using a Machine Learning Algorithm**, aimed to develop predictive models to assess the risk of loan defaults for the U.S. Small Business Administration (SBA) loan guarantee program. The dataset, spanning 1962–2014 with 899,164 observations and 29 variables, was used to inform lending decisions by minimizing credit losses and enhancing risk management.

**Submission Date**: April 18, 2025  
**Word Count**: 3,800 (Report)

The objective was to predict whether a loan would default (`MIS_Status = 0`) or not (`MIS_Status = 1`) using supervised machine learning models (Logistic Regression, Decision Tree, Random Forest, XGBoost) and evaluate the potential of unsupervised techniques (PCA, K-Means, Hierarchical Clustering) to enhance predictions. The project contributes to the SBA’s mission of fostering small business growth by reducing lending risks for banks.

## Dataset Description

The dataset, sourced from the U.S. SBA, contains historical loan data with 899,164 observations and 29 variables. Key features include:
- **Target Variable**: `MIS_Status` (0 = default, 1 = non-default)
- **Numerical Features**: `Term`, `DisbursementGross`, `SBA_Appv`, `NoEmp`, etc.
- **Categorical Features**: `NewExist`, `UrbanRural`, `RevLineCr`, `LowDoc`, `Industry`, etc.
- **Time Features**: `ApprovalDate`, `DisbursementDate`, `ChgOffDate`

The dataset was cleaned to 621,347 observations after preprocessing, with no missing values in critical variables, ensuring robust model development.

## Project Structure

The project is organized into four main tasks, as outlined in the assignment:

1. **Exploratory Data Analysis (EDA)**: Uncover patterns and insights to guide lending decisions.
2. **Logistic Regression Modeling**: Develop and evaluate a logistic regression model with varying probability thresholds.
3. **Alternative Machine Learning Models**: Compare Decision Tree, Random Forest, and XGBoost against Logistic Regression.
4. **Unsupervised Learning Techniques**: Assess PCA, K-Means, and Hierarchical Clustering for enhancing predictive modeling.

### Repository Contents

- **Group-4 (Code).ipynb**: Jupyter Notebook containing the complete code for data preprocessing, EDA, model training, evaluation, and unsupervised techniques.
- **Group-4 (Report).pdf**: Comprehensive project report detailing methodology, results, and recommendations.
- **README.md**: This file, providing an overview and instructions for the project.
- **Figures/**: Directory containing visualizations (e.g., correlation heatmaps, ROC curves, PCA biplots) generated during analysis.
- **Data/**: (Placeholder) Directory for the SBA dataset (not included due to size; available on Brightspace).

## Methodology and Key Findings

### 1. Exploratory Data Analysis (EDA)
- **Variable Categorization**: Variables were classified into Numerical Continuous (e.g., `Term`, `DisbursementGross`), Numerical Discrete (e.g., `NewExist`), and Time Categorical (e.g., `ApprovalDate`).
- **Correlation Analysis**: `Term` (0.26), `SBA_Appv` (0.10), and `GrAppv` (0.10) were top predictors of non-default.
- **Categorical Associations**: Chi-squared tests revealed significant associations with `Industry` (Construction, NAICS 23, ~25% default rate), `RevLineCr`, and `NewExist` (new businesses ~25% vs. existing ~15%).
- **Feature Importance**: Random Forest identified `Term` (0.35), `DaysToDisbursement` (0.15), and `DisbursementGross` (0.12) as key predictors.
- **Minority Variable**: Excluded from modeling due to regulatory compliance (Equal Credit Opportunity Act) and ethical concerns to avoid bias.

**Key Insight**: Longer loan terms and higher SBA-approved amounts correlate with lower default rates, while startups and construction industries are riskier.

### 2. Logistic Regression Modeling
- **Preprocessing**: Categorical variables were binary-encoded, numerical features standardized, and missing values handled (364,318 records post-cleaning).
- **Model Evaluation**: Trained on a 70-30 train-test split with thresholds (0.1, 0.2, 0.35, 0.5). Metrics included TPR, FPR, Accuracy, Precision, and F1 Score.
- **Results**:
  - Threshold 0.2 was optimal (TPR: 0.878, FPR: 0.360, AUC: 0.829), balancing default detection and false positives.
  - `Term` was the top predictor, followed by `NoEmp` and `RevLineCr`.
- **Cross-Validation**: 10-fold stratified CV yielded a mean ROC AUC of 0.829 (±0.02).
- **Comparison**: Using 11 selected variables (vs. 13) reduced overfitting while maintaining performance (AUC: 0.829).

**Recommendation**: Threshold 0.2 is recommended for balanced lending decisions.

### 3. Alternative Machine Learning Models
- **Models**: Decision Tree (max_depth=15), Random Forest (max_depth=20), XGBoost (max_depth=10).
- **Performance** (Threshold = 0.35):
  - **XGBoost**: AUC: 0.967, Accuracy: 93.4%, TPR: 83.6%, F1: 0.822
  - **Decision Tree**: AUC: 0.938, Accuracy: 93.0%, TPR: 83.3%, F1: 0.811
  - **Random Forest**: AUC: 0.961, Accuracy: 92.8%, TPR: 82.6%, F1: 0.805
  - **Logistic Regression**: AUC: 0.83, Accuracy: 85.0%, TPR: 60.0%, F1: 0.625
- **Key Observations**:
  - XGBoost outperformed all models, capturing complex non-linear patterns.
  - Decision Tree offered high interpretability, suitable for regulatory contexts.
  - Logistic Regression underperformed due to its linear assumptions.
- **Feature Importance**: `Term` dominated across models, with XGBoost highlighting `UrbanRural_Cat` and `RevLineCr_Cat`.

**Recommendation**: Deploy XGBoost (threshold = 0.35) for maximum predictive accuracy; use Decision Tree for interpretability in regulated environments.

### 4. Unsupervised Learning Techniques
- **Principal Components Analysis (PCA)**:
  - **Theoretical**: Reduces dimensionality by capturing variance but assumes linearity.
  - **Empirical**: Achieved AUC of 0.9753, surpassing XGBoost (0.967). However, 10 components were needed for 89.5% variance, reducing interpretability.
  - **Potential**: Enhances predictions by reducing noise but sacrifices transparency.
- **K-Means Clustering**:
  - **Theoretical**: Segments earners into clusters but assumes uniform, spherical groups.
  - **Empirical**: Optimal k=3 (Elbow Method), but imbalanced clusters (e.g., Cluster 0: 755,183 samples) and poor alignment with default status limited utility.
  - **Potential**: Minimal benefit due to imbalance and lack of predictive alignment.
- **Hierarchical Clustering**:
  - **Theoretical**: Flexible for non-spherical clusters but computationally intensive.
  - **Empirical**: Dendrogram suggested subgroups, but sampling constraints and weak default alignment reduced effectiveness.
  - **Potential**: Limited predictive value due to computational cost and noise sensitivity.

**Recommendation**: PCA offers slight predictive improvement but compromises interpretability. XGBoost with original features is preferred for practical lending.

## Installation and Usage

### Prerequisites
- Python 3.8+
- Libraries: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `xgboost`

### Setup
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/loan-default-prediction.git
   cd loan-default-prediction
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the Dataset**:
   - Obtain the SBA dataset from Brightspace (MyLearning > Group Assignment).
   - Place it in the `Data/` directory as `SBAnational_clean.csv`.

4. **Run the Code**:
   - Open `Group-4 (Code).ipynb` in Jupyter Notebook or Google Colab.
   - Update the dataset path in the code (e.g., replace `"C:/Users/adity/..."` with your path).
   - Execute cells sequentially to perform EDA, train models, and generate visualizations.

### Outputs
- **Visualizations**: Saved in the `Figures/` directory (e.g., `correlation_heatmap.png`, `roc_curve.png`).
- **Model Results**: Printed in the notebook (e.g., confusion matrices, AUC scores).
- **Report**: Refer to `Group-4 (Report).pdf` for detailed analysis and recommendations.

## Results and Recommendations

- **Best Model**: XGBoost (max_depth=10, threshold=0.35) achieved the highest performance (AUC: 0.967, Accuracy: 93.4%, F1: 0.822), recommended for advanced analytics in lending.
- **Alternative**: Decision Tree (max_depth=15) for regulatory transparency (AUC: 0.938, Accuracy: 93.0%).
- **Unsupervised Techniques**: PCA slightly improves AUC (0.9753) but reduces interpretability; K-Means and Hierarchical Clustering offer minimal benefits.
- **Fairness**: The `Minority` variable was excluded to comply with fair lending regulations (e.g., Equal Credit Opportunity Act) and avoid bias.

**Industry Implications**:
- Deploy XGBoost for accurate default prediction to minimize credit losses.
- Use Decision Tree in regulated environments for interpretability.
- Conduct cost-benefit analyses to fine-tune thresholds based on risk tolerance.
- Monitor fairness to ensure models do not disproportionately impact protected groups.

## Presentation

A 10-minute elevator pitch presentation was prepared for April 22, 2025, to pitch XGBoost as the preferred model for deployment. The presentation highlights its superior performance, practical implications, and alignment with the SBA’s risk management goals. Slides are available in the repository (to be added post-submission).

## References

1. Hand, D. J., & Henley, W. E. (1997). Statistical classification methods in consumer credit scoring: a review. *Journal of the Royal Statistical Society, Series A*, 160(3), 523–541.
2. Quinlan, J. R. (1986). Induction of decision trees. *Machine Learning*, 1(1), 81–106.
3. Breiman, L. (2001). Random forests. *Machine Learning*, 45(1), 5–32.
4. Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 785–794.
5. Abdi, H., & Williams, L. J. (2010). Principal component analysis. *Wiley Interdisciplinary Reviews: Computational Statistics*, 2(4), 433–459.
6. Equal Credit Opportunity Act, 15 U.S.C. § 1691 et seq., 1974.
7. Additional references are listed in the report.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

