import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib
from detect_gambling_comments import clean_comment_text, normalize_unicode  # your functions

# Step 1: Load preprocessed file
df = pd.read_csv('comments_labeled.csv', delimiter='¤', encoding='utf-8-sig')

# Step 2: Filter out uncertain labels
df = df[df['gambling'].isin([True, False])]

# Step 3: Preprocess comment text for ML
df['cleaned'] = df['comment'].apply(lambda x: clean_comment_text(normalize_unicode(x)))

# Step 4: Split
X_train, X_test, y_train, y_test = train_test_split(df['cleaned'], df['gambling'], test_size=0.2, random_state=42)

# Step 5: TF-IDF
vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=8000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Step 6: Train
model = LogisticRegression(max_iter=300)
model.fit(X_train_vec, y_train)

# Step 7: Evaluate
print(classification_report(y_test, model.predict(X_test_vec)))

# Step 8: Save
joblib.dump(model, 'gambling_model.joblib')
joblib.dump(vectorizer, 'vectorizer.joblib')
