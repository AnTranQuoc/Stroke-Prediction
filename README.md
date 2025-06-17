Stroke Prediction

## Description
As a simple practice exercise on the classification problem, it is expected that a person has a controlled risk disease or not.

## Features
### Stroke risk detection: The application uses PassiveAggressiveClassifier model using feed from available dataset.
### Dataset: The model has been trained on available dataset specifically for patients including 9 features.
### Use: The project is deployed as a web application using streamlit.

## Installation
1. Clone the project repository:
 `git clone`
 
2. Install the required dependencies:

   `pip install pandas numpy scikit-learn imbalanced-learn ydata-profiling lazypredict streamlit joblib`
 
      *Note: ydata_profiling can only be installed with python versions greater than 3.9 and less than 3.13
      In case you do not have the corresponding python version:

      `python -m venv .venv`
  
      `pip install -r requirements.txt`

## Usage
1. Start the Flask application:
 `streamlit run app_streamlit.py`
 
