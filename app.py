
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('punkt_tab') # Download the missing resource

# Step 1: Dataset Selection
# Assuming you have uploaded 'Fake.csv' and 'True.csv' to your Colab environment
try:
    fake_df = pd.read_csv('Fake.csv')
    true_df = pd.read_csv('True.csv')
    print("Datasets loaded successfully.")
except FileNotFoundError:
    print("Ensure 'Fake.csv' and 'True.csv' are uploaded to your Colab environment.")
    # You might want to provide instructions or an alternative here if files are missing.
    exit()

# Add a 'label' column to each dataframe
fake_df['label'] = 'fake'
true_df['label'] = 'real'

# Combine the datasets
df = pd.concat([fake_df, true_df], ignore_index=True)

# Drop the 'subject' and 'date' columns as they are not needed for this project
df = df.drop(columns=['subject', 'date'], errors='ignore')

# Step 2: Model Implementation
# A. Data Preparation

# Handle duplicates
initial_rows = len(df)
df.drop_duplicates(subset=['title', 'text'], inplace=True)
print(f"Removed {initial_rows - len(df)} duplicate rows.")

# Handle missing values
print(f"Missing values before handling:
{df.isnull().sum()}")
df.fillna('', inplace=True) # Fill empty strings for text/title if any
print(f"Missing values after handling:
{df.isnull().sum()}")

# Combine title and text for processing
df['full_text'] = df['title'] + ' ' + df['text']

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    # Lowercase the text
    text = text.lower()
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    # Tokenization and Lemmatization
    tokens = nltk.word_tokenize(text)
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(lemmatized_tokens)

print("
Starting text preprocessing...")
df['processed_text'] = df['full_text'].apply(preprocess_text)
print("Text preprocessing complete.")

# B. Convert text into vectors using TF-IDF
tfidf_vectorizer = TfidfVectorizer(max_features=5000) # You can adjust max_features
X = tfidf_vectorizer.fit_transform(df['processed_text'])
y = df['label']

# C. Model Training
# Split into train/test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"
Training data shape: {X_train.shape}")
print(f"Testing data shape: {X_test.shape}")

# Train a Logistic Regression classifier on TF-IDF features
print("Training Logistic Regression model...")
model = LogisticRegression(max_iter=1000) # Increased max_iter for convergence
model.fit(X_train, y_train)
print("Model training complete.")

# Make predictions
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)

# Evaluate with:
# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"
Accuracy: {accuracy:.4f}")

# Precision, Recall, F1-score
precision = precision_score(y_test, y_pred, pos_label='fake')
recall = recall_score(y_test, y_pred, pos_label='fake')
f1 = f1_score(y_test, y_pred, pos_label='fake')
print(f"Precision (Fake): {precision:.4f}")
print(f"Recall (Fake): {recall:.4f}")
print(f"F1-score (Fake): {f1:.4f}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred, labels=['real', 'fake']) # Specify labels for better interpretation

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

# 5. User Interaction (Optional Streamlit - for Jupyter, we'll simulate it)
print("
--- User Interaction Simulation ---")

def predict_news(article_text):
    processed_article = preprocess_text(article_text)
    vectorized_article = tfidf_vectorizer.transform([processed_article])
    prediction = model.predict(vectorized_article)[0]

    # Get confidence probability
    probabilities = model.predict_proba(vectorized_article)[0]

    if prediction == 'fake':
        confidence = probabilities[model.classes_ == 'fake'][0]
    else:
        confidence = probabilities[model.classes_ == 'real'][0]

    return prediction, confidence

# Example usage:
sample_article_fake = "Hillary Clinton calls for 'civil discussion' in a campaign event but her words were twisted by a right-wing media outlet to suggest she advocates violence."
sample_article_real = "The economy grew by 0.3% in the last quarter, slightly exceeding analyst expectations, according to government reports released today."
sample_article_new = "Scientists discover new planet made entirely of cheese in a galaxy far, far away, defying all known laws of physics."

print("
Processing sample article 1 (likely fake):")
pred, conf = predict_news(sample_article_fake)
print(f"Prediction: {pred.upper()} (Confidence: {conf:.2f})")

print("
Processing sample article 2 (likely real):")
pred, conf = predict_news(sample_article_real)
print(f"Prediction: {pred.upper()} (Confidence: {conf:.2f})")

print("
Processing sample article 3 (new, likely fake):")
pred, conf = predict_news(sample_article_new)
print(f"Prediction: {pred.upper()} (Confidence: {conf:.2f})")

# Let's generate a quick visualization to represent the fake news scenario.
print("
Here's a visual representation of a 'Fake News Detector' in action:")
