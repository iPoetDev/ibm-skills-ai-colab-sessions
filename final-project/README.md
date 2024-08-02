# **FINAL PROJECT**: IBM Programme for Artifical Intelligence 2024:<br> **Credit Card Fraud Detection** <sup><small>Graded, Assessed</small></sup>

## Introduction

**<ins>The Final Project</ins>**

The Final Project of IBM SkillBuild Programme for Artificial Intelligenced 2024 deals with incidence of fraud by training and evaluating of selected machine learning models and algorithms on a highly unbalanced dataset of credit card transactions. The goal is to identify which transactions are fraudulent (labeled as 1) and which are legitimate (labeled as 0).

**<ins>DataSet</ins>**

The dataset, as prepared by [@leborgne2022fraud](references.bib) of Worldline and the Machine Learning Group (http://mlg.ulb.ac.be) of ULB (Université Libre de Bruxelles, has been collected and analysed in a research collaboration on big data mining and fraud detection.

*Contains:*
- Transactions made by credit cards in September 2013 by European cardholders.
- Anonymized credit card transactions labeled as fraudulent or genuine

**<ins>Objective</ins>**

To apply the theory, utilise the hands-on live techical sessions examples in the implemenation of:

> 1️⃣ Credit Card Fraud Detection
>> *  Use python libraries and frameworks
>> *  Apply a workflow breakdown structure to the project, step by step.
>> *  Gather and discet a kaggle dataset, as provided.
>> *  Create an orginal authentic solution to the problem<br>

### Audience

- Technical: Data Scientists, AI Engineers, Data Analysts, AI Specialists
- Non-Technical: Those with industry or academic experince in Credit Card Fraud; AI Product Owners and AI line of business Professionals.

### The Goal

> #### The goal is to identify which transactions are fraudulent (labeled as 1) and which are legitimate (labeled as 0).

> <hr>

### Getting Started

For a quick start, open the notebook on Google CoLab, to inspect the solution

[![GoogleColab](https://img.shields.io/badge/Google%20CoLab:--0e80c1?logo=googlecolab&logoColor=d39816)](https://colab.research.google.com/github/iPoetDev/ibm-skills-ai-colab-sessions/blob/final-project/final-project/IBMSkillsBuild_ANSWER_AI_CreditCardFraudDetection_EN.ipynb#scrollTo=ead4329e "Open in Google CoLab: Final Project: Credit Card Fraud Detection")

## Table of Contents

- [Introduction]()
  - [Getting Started]()
- [Kaggle Dataset]()
- [Research]()
- [Solution]()
- [Findings]()
- [Take Aways]()
- [Summary]()

## Kaggle Dataset

This dataset is a valuable resource for studying credit card fraud detection using machine learning. The the imbalanced nature of the dataset presents an additional challenge that must be addressed when building and evaluating fraud detection models.Confidentiality of data set is protected by PCA transformation of original features; This is because these may contain sensitive information about the credit card holders or the transactions themselves

**<ins>Context</ins>**

It is important that credit card companies are able to recognize fraudulent credit card transactions so that customers are not charged for items that they did not purchase.

**<ins>CreditCard.csv</ins>**

- **`Time`**: 2 Days
- **`Transactions`**: 284,807 transactions
- **`Incidents`**: 492 frauds
- **`Positives`**: 0.172% of all transactions.

**<ins>Download</ins>**
<details>
<summary>CreditCard.csv: Local</summary>
<a href="assets/datasets/creditcard.csv"><b><ins>Download</ins></b>: &nbsp; <sub><img src="https://img.shields.io/badge/Kaggle%20Download-CSV:%20Credit%20Card%20Fraud-20beff?logo=kaggle&logoColor=w20beff" alt="Kaggle"></sub></a> <sup><ins>x</ins></sup>
</details>


**<ins>Utility</ins>**
<details>
<summary>Kaggle Score Card</summary> 
<img src="assets/img/Kaggle_usability_card.png"> <sup><ins>x</ins></sup>
</details>

- <sup><ins>x</ins></sup> Kaggle (2018). [": Download".](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud/data). Last Acecss: Aug 2024. https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud/data
- <sup><ins>x</ins></sup> Kaggle (2018). ["Credit Card Fraud Detection: Utility"](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud/data). Last Acecss: Aug 2024. https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud/data

### Highlights

- No details are provided on the original features and more background information about the data due to confidentiality issues.
- Contains information about credit card transactions, where each row represents a single transaction. 
- The dataset has 31 features (columns).
- Principal Component Analysis (PCA) techniques were applied to the original features into a new set of features that are uncorrelated with each other.
- Time, Ammount and Class where not transformed, and are left as-is.
- Fraudulent transactions are labeled **`1`**. legitimate are labeled **`0`**.

### Methods & Aspects

 The use of PCA to transform the original features ensures confidentiality while still preserving the important information needed for fraud detection.

#### Methods: PCA Transformation

- Is used to transform the original features into a new set of features called principal components. 
- These principal components are
    - Linear combinations of the original features.
    - Chosen in a way that maximizes the amount of variance in the data that is captured by each component.
- Due to confidentiality concerns, PCA was selected as:
    - As the original features may contain sensitive information about the credit card holders or the transactions themselves.
    - By transforming the features using PCA, the original data is obscured while still preserving the important information needed for fraud detection.

#### Aspect: Imbalanced Dataset

- An imbalanced datasets means that there are many more legitimate transactions than fraudulent ones. 
  - This is common in real-world fraud detection scenarios.
  - Fraudulent transactions are relatively rare compared to legitimate ones.

#### Challenge: Imbalanced Dataset

> The goal is to identify which transactions are fraudulent (labeled as 1) and which are legitimate (labeled as 0).

- Dealing with imbalanced datasets is an important challenge in machine learning.
- Requires special techniques to ensure that the model performs well on both classes.

## Research

The Kaggle dataset's authors, [@leborgne2022fraud](references.bib), Yann-Aël Le Borgne, Gianluca Bontempi produced the ["Reproducible machine Learning for Credit Card Fraud Detection - Practical Handbook"](https://fraud-detection-handbook.github.io/fraud-detection-handbook/Foreword.html) and on GitHub  <sub>[![GitHub](https://img.shields.io/badge/leborgne2022fraud-Fraud--Detection--Handbook-181717?logo=github&logoColor=white)](https://github.com/Fraud-Detection-Handbook/fraud-detection-handbook "GitHub.com: Yann-Aël Le Borgne, Gianluca Bontempi 'Reproducible machine Learning for Credit Card Fraud Detection - Practical Handbook'")</sub>. By publisher: [Machine Learning Group (Université Libre de Bruxelles - ULB)](https://mlg.ulb.ac.be/wordpress/) and [Worldline](https://worldline.com/).

## Solution

### Project Management

**<ins>Open Projects</ins>**<br><br>
**<ins>[Open:](https://github.com/users/iPoetDev/projects/22/views/1)</ins>**<sub> &nbsp;&nbsp; [![GitHub](https://img.shields.io/badge/GitHub%20Projects%20by%20iPoetDev:-Credit%20Card%20Fraud-181717?logo=github&logoColor=white)](https://github.com/users/iPoetDev/projects/22/views/1 "PROJECT: IBM Programme for Artificial Intelligence - Credit Card Fraud")</sub>

- Project control and iterative/agile workflows to show case the work items and plan for solving final Project for the Programme for Artificial Intelligence 2024.

### The Analysis

To view the solution in its orignal unanalysed form and in its final form, click on each button below.

| Notebook \| GitHub | **<ins>Source</ins>** | ___ | **<ins>Answer</ins>** | Date Submitted | 
| :--- | :--- | :--- | :--- | :--- | 
| [CreditCardFraudDetection_EN.ipynb](IBMSkillsBuild_ANSWER_AI_CreditCardFraudDetection_EN.ipynb) | [![Google Colab](https://img.shields.io/badge/Google%20Colab-CreditCardFraudDetection_EN.ipynb-F9AB00?logo=googlecolab&logoColor=white)](https://colab.research.google.com/github/iPoetDev/ibm-skills-ai-colab-sessions/blob/final-project/final-project/IBMSkillsBuild_SOURCE_AI_CreditCardFraudDetection_EN.ipynb "Open in Google CoLab: EMPTY Credit Card Fraud Detection") | ___ | [![Google Colab](https://img.shields.io/badge/Google%20Colab-Credit_Card_Fraud_Detection_EN.ipynb-darkred?logo=googlecolab&logoColor=white)](https://colab.research.google.com/github/iPoetDev/ibm-skills-ai-colab-sessions/blob/final-project/final-project/IBMSkillsBuild_ANSWER_AI_CreditCardFraudDetection_EN.ipynb#scrollTo=ead4329e "Open in Google CoLab: Final Project: Credit Card Fraud Detection") | 2024-08-dd | 

<br><sup><ins>`Opens in Google Colabatory`</ins></sup>

## Findings

### Analysis

**<ins>Question 1</ins>**: What is the percentage of fraud transactions in the dataset? <sup>*(% percentage)*</sup>

- 

**<ins>Question 2</ins>**: What is the average transaction amount for fraud transactions? <sup>*(mean/average)*</sup>)

- 

### Visualisations

**<ins>Question 1</ins>**: How many fraud transactions are there compared to non-fraud transactions? <sup>*(A bar plot)*</sup>

-

**<ins>Question 2</ins>**: What is the distribution of transaction amounts for fraud transactions? <sup>*(A histogram)*</sup>)

-

### Model Evaluations

**<ins>Hyperparameters</ins>**


**<ins>Predictitions</ins>**


**<ins>Performance & Metrics</ins>**


**<ins>Accuracy</ins>**

## Take Aways

**<ins>Key Skills</ins>**


**<ins>Key Learnings</ins>**


**<ins>___</ins>**

# Summary

<hr>

>> <hr>

<hr>

## References

## Author

## ChangeLog