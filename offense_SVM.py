# Imported Libraries
import csv
import pandas as pd
import re
import string
import nltk
import sklearn
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import precision_recall_fscore_support as prfs

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

#Read Dataset
df = pd.read_csv('olid-training-v1.0.tsv',sep = '\t')

#Clean Data
def clean_text(text):
    text = text.lower()  # Convert to lowercase
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    text = ' '.join([word for word in text.split() if word not in stop_words])  # Remove stopwords
    tokens = word_tokenize(text)  # Tokenize text
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]  # Lemmatize
    return " ".join(tokens)

df = df[['id', 'tweet', 'subtask_a','subtask_b','subtask_c']]
df['tweet'] = df['tweet'].astype(str).apply(clean_text)

#Split data into training and test data
X_train, X_test, y_train, y_test = train_test_split(df['tweet'], df['subtask_a'], test_size=0.2, random_state=42)

# Text processing and SVM pipeline
pipeline = Pipeline([
    ("vectorizer", TfidfVectorizer(stop_words="english", max_features=5000)),
    ("classifier", SVC(kernel="linear",probability=True))
])

#Train first model
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

#Print model scores
print("Offensive Identification Report:")
print("Accuracy:", accuracy_score(y_test, y_pred))
p_micro, r_micro, f1_micro, _ = prfs(y_pred=y_pred, y_true=y_test, average="micro")
print("Micro Evaluation: Precision: %.4f; Recall: %.4f; F1-Score: %.4f" % (p_micro, r_micro, f1_micro))

# Train second model (targeted vs untargeted) on offensive tweets
offensive_tweets = df[df["subtask_a"] == 'OFF']
X_target_train, X_target_test, y_target_train, y_target_test = train_test_split(
    offensive_tweets["tweet"], offensive_tweets["subtask_b"], test_size=0.2, random_state=42
)

# Pipeline for second model
pipeline_target = Pipeline([
    ("vectorizer", TfidfVectorizer(stop_words="english", max_features=5000)),
    ("classifier", SVC(kernel="linear",probability=True))
])

# Train second model
pipeline_target.fit(X_target_train, y_target_train)
y_target_pred = pipeline_target.predict(X_target_test)

#Print Second model scores
print("Targeted Report:")
print("Accuracy:", accuracy_score(y_target_test, y_target_pred))
p_micro, r_micro, f1_micro, _ = prfs(y_pred=y_target_pred, y_true=y_target_test, average="micro")
print("Micro Evaluation: Precision: %.4f; Recall: %.4f; F1-Score: %.4f" % (p_micro, r_micro, f1_micro))

# Train third model (Target type) on targeted tweets
targeted_tweets = df[df["subtask_b"]=='TIN']
X_type_train, X_type_test, y_type_train, y_type_test = train_test_split(
    targeted_tweets["tweet"], targeted_tweets["subtask_c"], test_size=0.2, random_state=42
)

# Pipeline for third model
pipeline_type = Pipeline([
    ("vectorizer", TfidfVectorizer(stop_words="english", max_features=5000)),
    ("classifier", SVC(kernel="linear", probability=True))
])

#Print Third model scores
pipeline_type.fit(X_type_train, y_type_train)
y_type_pred = pipeline_type.predict(X_type_test)
print("Target Type Report")
print("Accuracy:", accuracy_score(y_type_test, y_type_pred))
p_micro, r_micro, f1_micro, _ = prfs(y_pred=y_type_pred, y_true=y_type_test, average="micro")
print("Micro Evaluation: Precision: %.4f; Recall: %.4f; F1-Score: %.4f" % (p_micro, r_micro, f1_micro))