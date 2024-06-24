# meds-interp
meds-interp is a Python library designed to facilitate the interpretation of embeddings derived from Electronic Health Record (EHR) data. It includes helper functions that enable the visualization and analysis of these embeddings through various clustering techniques.

Features
1. Linear Probing Analysis
Ensemble Embeddings: This feature allows for the combination of multiple embeddings to enhance the robustness and accuracy of the linear probing analysis.
Automated Hyperparameter Tuning: Automates the search for optimal hyperparameters for logistic and linear regression models, as well as for ensemble methods, ensuring the best possible analysis outcomes.
2. KMeans Clustering
Integration with Faiss Library: Utilizes the Faiss library for efficient clustering of large datasets.
Ensemble Embeddings: Similar to linear probing, this feature supports the ensembling of multiple embeddings to improve clustering results.
Automated Hyperparameter Tuning: Facilitates automated searching for the best logistic and linear regression settings and ensemble configurations.
3. Cluster Interpretation
Code Prevalence Analysis: Interprets clusters based on the prevalence of specific codes within them. Codes are considered prevalent in two primary ways:
Distribution Level Prevalence: Identifies codes that exhibit significant deviation in their distribution within a cluster compared to the general population.
Getting Started
To get started with meds-interp, follow the installation and setup instructions below. [Instructions to be added by the maintainer.]
