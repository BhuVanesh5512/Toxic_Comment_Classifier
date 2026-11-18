# Toxic Comment Classifier

An NLP and machine learning project that classifies a text comment as **toxic** or **non-toxic**. The project uses TF-IDF text features, manually engineered toxicity-word features, and an XGBoost classifier. A Streamlit application provides a simple interface for trying the trained model.

## Features

- Text preprocessing: lowercasing, punctuation and number removal, stopword removal, and non-ASCII character cleanup.
- TF-IDF features with unigrams and bigrams (up to 1,500 features).
- Additional word-count features for profane words and toxicity-related terms, including insults, hate speech, threats, harassment, and offensive language.
- XGBoost binary classification.
- Streamlit interface for classifying user-entered comments.

## Model Performance

The metrics below are taken from the evaluation output saved in `model.ipynb`. The model was evaluated on a stratified 80/20 train-test split of 10,000 rows (2,000 test examples).

| Metric | Score |
|---|---:|
| Accuracy | **94.2%** |
| Toxic-class F1-score | **0.38** |
| Non-toxic-class F1-score | **0.97** |
| Macro-average F1-score | **0.67** |
| Weighted-average F1-score | **0.94** |

### Classification report

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| Non-toxic (0) | 0.97 | 0.97 | 0.97 | 1,915 |
| Toxic (1) | 0.35 | 0.41 | 0.38 | 85 |
| Macro average | 0.66 | 0.69 | 0.67 | 2,000 |
| Weighted average | 0.95 | 0.94 | 0.94 | 2,000 |

**Note:** The test set is imbalanced: only 85 of the 2,000 test comments belong to the toxic class. Therefore, accuracy and weighted F1 alone do not fully represent toxic-comment detection performance. The toxic-class F1-score (0.38) is especially important when assessing this model.

## Project Files

- `model.ipynb` — preprocessing, feature engineering, model training, evaluation, and model export.
- `app.py` — Streamlit application.
- `test.py` — alternate/earlier Streamlit testing script.
- `tfidf_vectorizer.pkl` — saved TF-IDF vectorizer.
- `xgboost_model.pkl` — saved XGBoost classifier.
- `requirements.txt` — Python dependencies.

## Setup and Run

1. Clone the repository:

   ```bash
   git clone https://github.com/BhuVanesh5512/Toxic_Comment_Classifier.git
   cd Toxic_Comment_Classifier
   ```

2. (Recommended) Create and activate a virtual environment.

3. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Download the NLTK English stopwords resource:

   ```bash
   python -m nltk.downloader stopwords
   ```

5. Start the Streamlit application:

   ```bash
   streamlit run app.py
   ```

The app loads the saved vectorizer and classifier from the repository and lets you enter a comment to receive a toxic/non-toxic prediction.

## Training

To retrain the model, place a `train.csv` file in the project directory with the columns:

- `comment_text` — the comment text.
- `target` — the toxicity label (values greater than 0.5 are converted to toxic class 1; other values become class 0).

Then run `model.ipynb`. The notebook currently loads up to 10,000 rows, creates a stratified 80/20 train-test split, trains the XGBoost model, prints its evaluation metrics, and saves the vectorizer and classifier.

## Tech Stack

Python · Pandas · NLTK · Scikit-learn · TF-IDF · XGBoost · SciPy · Streamlit
