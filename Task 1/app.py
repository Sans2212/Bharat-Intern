import streamlit as st
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

# Load the dataset
df = pd.read_csv('spam.csv', encoding='latin-1')
df = df.drop(["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis=1)
df = df.rename(columns={"v1":"label", "v2":"text"})

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)

# Create a pipeline with TF-IDF vectorizer and SVM classifier
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', SVC(kernel='linear'))
])

# Train the model
pipeline.fit(X_train, y_train)

# Evaluate the model
accuracy = accuracy_score(y_test, pipeline.predict(X_test))
st.write("Model Accuracy:", accuracy)

# Now doing Streamlit coding
st.title("SMS Spam/Ham Classifier")
input_sms = st.text_area("Enter message:")

if st.button('Predict'):
    # Predict
    result = pipeline.predict([input_sms])[0]
    st.write("Your message is:", result)
