# 🧠 Sentiment Analysis on 1.6M Tweets using SVM 🐦

This project focuses on classifying tweets as **positive** or **negative** using **Support Vector Machine (SVM)** with **TF-IDF features**. The dataset used is large-scale and originally imbalanced, requiring preprocessing and visualization techniques to enhance model performance and interpretability.

---

## 📂 Dataset

* **Source**: [Kaggle - Sentiment140](https://www.kaggle.com/datasets/kazanova/sentiment140)
* **Description**:
  Contains **1.6 million** labeled tweets. Each tweet is labeled with a sentiment:

  * `0` = Negative
  * `4` = Positive (converted to 1 for binary classification)
    After balancing:
  * **800,000 positive**, **800,000 negative**

---

## ⚙️ Technologies Used

* Python
* Pandas, NumPy
* Matplotlib, Seaborn
* Scikit-learn
* WordCloud
* Jupyter Notebook

---

## 🔍 Project Workflow

### 1. Data Loading & Initial Exploration

```python
import pandas as pd
df = pd.read_csv('Sentiment Analysis.csv', encoding='latin-1', header=None)
df.columns = ['Target', 'ID', 'Date and Time', 'Search Word', 'Username', 'Tweets']
df['Target'] = df['Target'].replace(4, 1)
```

### 2. Text Preprocessing

* Removed:

  * Mentions (@user)
  * URLs
  * RT tags
  * Punctuation
  * Hashtags (symbol only)
* Converted text to lowercase

```python
import re
def clean_text(text):
    text = re.sub(r'@[A-Za-z0-9_]+', '', text)
    text = re.sub(r'https?:\/\/\S+', '', text)
    text = re.sub(r'RT[\s]+', '', text)
    text = re.sub(r'#', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text.lower()

df['Clean Tweets'] = df['Tweets'].apply(clean_text)
df = df[['Clean Tweets', 'Target']]
```

---

## 📊 Exploratory Data Analysis (EDA)

### ✅ Class Distribution

```python
sns.countplot(x='Target', data=df)
```

### 📝 Tweet Length Distribution

```python
df['tweet_length'] = df['Clean Tweets'].apply(len)
sns.histplot(data=df, x='tweet_length', bins=50, kde=True)
```

### ☁️ Word Clouds

```python
# show_wordcloud(1) for Positive
# show_wordcloud(0, color='black') for Negative
```

### 🔠 Top 20 Most Common Words

```python
from collections import Counter
word_freq = Counter(" ".join(df['Clean Tweets']).split())
common_df = pd.DataFrame(word_freq.most_common(20), columns=['word', 'count'])
sns.barplot(x='count', y='word', data=common_df)
```

### 📦 Word Count by Sentiment

```python
df['word_count'] = df['Clean Tweets'].apply(lambda x: len(x.split()))
sns.boxplot(x='Target', y='word_count', data=df)
```

---

## 🧠 Model Training

### 1. Data Preparation

```python
from sklearn.model_selection import train_test_split
X = df['Clean Tweets']
y = df['Target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

### 2. TF-IDF Vectorization

```python
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)
```

### 3. Training SVM Classifier

```python
from sklearn.svm import LinearSVC
model = LinearSVC(dual=True)
model.fit(X_train_tfidf, y_train)
```

### 4. Evaluation

```python
from sklearn.metrics import classification_report
y_pred = model.predict(X_test_tfidf)
print(classification_report(y_test, y_pred))
```

#### 📈 Results:

| Metric       | Negative (0) | Positive (1) |
| ------------ | ------------ | ------------ |
| Precision    | 0.80         | 0.78         |
| Recall       | 0.77         | 0.81         |
| F1-Score     | 0.79         | 0.79         |
| **Accuracy** | **0.79**     | -            |

---

## 🛠️ Hyperparameter Tuning

Used `GridSearchCV` to test different `C` values:

```python
from sklearn.model_selection import GridSearchCV
param_grid = {'C': [0.01, 0.1, 1, 10]}
grid = GridSearchCV(LinearSVC(dual=True), param_grid, cv=3)
grid.fit(X_train_tfidf, y_train)
print("Best C:", grid.best_params_)
```

**Note**: Results were consistent before and after tuning.

---

## 💾 Model Export

```python
import joblib
joblib.dump(model, "sentiment_svm_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
```

---

## 🔮 Real-Time Prediction

```python
tweet = ["I love this product!"]
tweet_tfidf = vectorizer.transform(tweet)
prediction = model.predict(tweet_tfidf)
print("Prediction:", prediction[0])  # Output: 1 (Positive)
```

---

## ✅ Conclusion

* The **SVM model with TF-IDF features** performed effectively on a large tweet dataset.
* Achieved a balanced F1-score of **0.79** for both classes.
* Preprocessing and visualization played a key role in understanding the data.
* The model is now ready for real-world tweet sentiment prediction.




