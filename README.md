# Fake News Detection using Tensorflow (and Logistic Regression with TFIDF)

Author: Rohan Mahajan

## Overview

This project focuses on detecting fake news articles using two different approaches: a deep learning model built with TensorFlow and logistic regression with TF-IDF (Term Frequency-Inverse Document Frequency) vectorization. The goal is to compare the performance of these two methods for fake news detection.

## Project Structure

The project is divided into several main steps:

1. **Importing Data and Initial Pre-Processing**:
   - Load the dataset containing news articles.
   - Remove unnecessary columns.
   - Handle missing values.
   - Remove duplicate articles.
   - Encode labels (0 for FAKE, 1 for REAL).

2. **Tokenizing and Further Cleaning Data**:
   - Tokenize and preprocess the text data.
   - Remove stopwords and non-letter words.
   - Pad sequences to make them uniform in length.

3. **Generating Word Embeddings**:
   - Use pre-trained word embeddings (GloVe) to represent words.
   - Create an embeddings matrix for the model.

4. **Creating, Fitting, and Testing TensorFlow Model**:
   - Split the data into training and testing sets.
   - Build a deep learning model using TensorFlow.
   - Train the model on the training data.
   - Evaluate the model on the testing data.

5. **Visualize Results**:
   - Analyze the model's performance using visualizations (optional).

6. **Train and Test Logistic Regression Model with TF-IDF**:
   - Implement a logistic regression model using TF-IDF vectorization.
   - Train and test the model.

7. **Conclusion**:
   - Summarize the findings and insights from both models.
   - Discuss the performance and limitations of each approach.

## Getting Started

To run this project, you will need:

- Python (>=3.6)
- TensorFlow
- scikit-learn
- NLTK
- regex
- matplotlib
- pandas
- numpy

You can install the required packages using pip:

```
pip install tensorflow scikit-learn nltk regex matplotlib pandas numpy
```

## Usage

1. Clone the repository to your local machine.

```
git clone https://github.com/your-username/fake-news-detection.git
```

2. Navigate to the project directory.

```
cd fake-news-detection
```

3. Execute the Jupyter Notebook or Python scripts to run the project.

```
jupyter notebook Fake_News_Detection.ipynb
```

4. Follow the instructions and code comments in the notebook or scripts to explore and analyze the fake news detection models.

## Results

The project aims to provide insights into the effectiveness of both deep learning and traditional machine learning approaches for fake news detection. You will find detailed results and performance metrics in the project's notebook or output logs.

## Acknowledgments

- The project uses pre-trained GloVe word embeddings, which were created by the Stanford NLP Group.
