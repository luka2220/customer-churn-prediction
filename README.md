# Customer Churn Prediction with Machine Learning

## Setup and Installation Documentation

* All the application code lives in the root directory
* The machine learning models are serialized in python pickle files under `model-binaries`
* The data set is `churn.csv`

### Running with a python virtual environment

* NOTE: Make sure python is installed on your system

*Mac/Linux Setup*

- `python -m .venv`
- `source .venv/bin/activate`
- `python -m pip install -r requirements.txt`
- `create the .env file with secret to the GROQ API`
- `python -m streamlit run main.py`

*Windows Setup*

- `python -m venv .venv`
- `.\.venv\Scripts\activate`
- `pip install requirements.txt`
- `create the .env file with secret to the GROQ API`
- `python -m streamlit run main.py`


