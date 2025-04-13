
# LiverDiseaseSuperLearner

**Developing a semi-supervised learning model on Liver-Disease dataset using a super learner and deploying the model on other datasets.**

---

## Overview

This project implements a semi-supervised learning pipeline for liver-disease prediction. The model leverages a super learner approach that integrates multiple base classifiers and a meta learner to achieve high prediction accuracy. The system is designed with object-oriented programming principles to ensure modularity, maintainability, and scalability.

---

## Project Workflow

The project workflow is divided into the following five major steps:

1. **Data Pre-processing Phase**
   - **Outlier Removal:** Clean outliers from all columns.
   - **Missing Values Imputation:** Impute missing values in all columns.
   - **Normalization:** Normalize all columns to ensure uniform scale across features.

2. **Unsupervised Learning for Generating Labels**
   - **K-means Clustering:** Apply K-means clustering on three selected features (for example, LiverFunctionTest, BMI, and Age) to partition the dataset into two clusters.
   - **Label Assignment:** Assign the label *Liver-Disease* to the cluster with a higher average LiverFunctionTest and *No Liver-Disease* to the other cluster.
   - **Outcome Column:** Add a new column (`Outcome`) to the dataset containing `1` for Liver-Disease and `0` for No Liver-Disease to be used as classification labels.

3. **Feature Extraction**
   - **Train-Test Split:** Split the dataset into training and testing sets (80% training, 20% testing).
   - **Dimensionality Reduction:** Use Principal Component Analysis (PCA) on the training data to generate 3 new principal components from the existing features (excluding the outcome).
   - **Data Transformation:** Transform both the training and test sets into the new PCA space.

4. **Classification Using a Super Learner**
   - **Base Classifiers:** Define three base classifiers – Naïve Bayes, Neural Network, and K-Nearest Neighbors (KNN).
   - **Meta Learner:** Define a Decision Tree as the meta learner.
   - **Training with Cross-validation:** Train the meta learner on the outputs of the base classifiers using 5-fold cross validation.
   - **Hyperparameter Tuning:** Find the optimal hyperparameters for all models to achieve the best accuracy.
   - **Accuracy Reporting:** Evaluate and report the model accuracy on the test set.

5. **Model Deployment on Other Datasets**
   - **Generalization:** Adapt the code with minor changes (e.g., encoding categorical variables) to apply the model pipeline (Steps 1, 3, and 4) on external datasets.
   - **Outcome Usage:** Use the last column of the new dataset as the outcome label and calculate the accuracy of the deployed model.

---

## Project Structure

```plaintext
LiverDiseaseSuperLearner/
├── data/                   # Data files for training/testing and external datasets
├── src/
│   ├── __init__.py
│   ├── preprocessing.py    # Data cleaning, imputation, normalization
│   ├── clustering.py       # K-means clustering and label generation
│   ├── feature_extraction.py  # PCA and train-test split
│   ├── models.py           # Base classifiers and meta learner definitions
│   ├── super_learner.py    # Super learner model training and evaluation
│   └── deployment.py       # Code for applying the model on external datasets
├── tests/                  # Unit tests for the modules
├── README.md               # Project overview and instructions
└── requirements.txt        # Project dependencies
```

---

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yourusername/LiverDiseaseSuperLearner.git
   cd LiverDiseaseSuperLearner
   ```

2. **Install the required packages:**

   ```bash
   pip install -r requirements.txt
   ```

   *Dependencies include popular libraries such as scikit-learn, pandas, numpy, and others as needed.*

---

## Usage

### Running the Full Pipeline

To run the complete pipeline from pre-processing to model evaluation:

```bash
python src/super_learner.py
```

### Deploying the Model on a New Dataset

Ensure that the new dataset is formatted similarly and placed in the `data/` folder, then run:

```bash
python src/deployment.py --dataset new_dataset.csv
```

*Additional arguments and configurations are available in the code documentation.*

---

## Object-Oriented Design

The project is implemented using object-oriented programming (OOP) principles. Key benefits include:

- **Modularity:** Each phase of the pipeline (data processing, feature extraction, modeling, deployment) is encapsulated in its own class or module.
- **Maintainability:** The clear module separation makes it easier to modify or extend parts of the system.
- **Scalability:** New features, such as additional classifiers or enhanced data pre-processing techniques, can be easily integrated.

---

## Contributing

Contributions are welcome! Please create a pull request or open an issue if you have suggestions or improvements.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
