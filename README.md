# meds-interp

**meds-interp** is a Python library designed to facilitate the interpretation of embeddings derived from Electronic Health Record (EHR) data. It includes helper functions that enable the visualization and analysis of these embeddings through various clustering techniques.

## Features

1. **Linear Probing Analysis**
   - **Ensemble Embeddings:** Allows for the combination of multiple embeddings to enhance the robustness and accuracy.
   - **Automated Hyperparameter Tuning:** Automates the search for optimal hyperparameters for logistic and linear regression models, as well as for ensemble methods.

2. **KMeans Clustering & KNN Classifier/Regression**
   - **Integration with Faiss Library:** Utilizes the Faiss library for efficient clustering of large datasets.
   - **Ensemble Embeddings:** Supports the ensembling of multiple embeddings to improve clustering and prediction results.
   - **Automated Hyperparameter Tuning:** Facilitates automated searching for the best settings and configurations.

3. **Cluster Interpretation**
   - **Code Prevalence Analysis:** Interprets clusters based on the prevalence of specific codes within them, focusing on:
     - **Distribution Level Prevalence:** Identifies codes that exhibit significant deviation in their distribution within a cluster compared to the general population.

## Getting Started

To get started with `meds-interp`, install the library using the following command:

```console
pip install .
