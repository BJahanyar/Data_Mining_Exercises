import hazm 
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from io import StringIO
from string import punctuation
from sklearn.svm import LinearSVC
from IPython.display import display
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_predict
from sklearn.datasets import make_classification
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import chi2
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

 بارگزاری اطلاعات
df = pd.read_csv('AllComments.csv')

# حذف اطلاعات تکراری
df = df[pd.notnull(df['Label'])]
df = df[pd.notnull(df['comment'])]
df = df.drop_duplicates()

# 'category_id' ایجاد یک ستون به نام 
df2 = df.copy()
df2['category_id'] = df2['Label'].factorize()[0]
category_id_df = df2[['Label', 'category_id']].sort_values('category_id')

# ایجاد یک دیکشنری
category_to_id = dict(category_id_df.values)
id_to_category = dict(category_id_df[['category_id', 'Label']].values)

# ایجاد یک نمود فراوانی تکرار ایموجی‌ها
fig = plt.figure(figsize=(8,6))
df2.groupby('Label').comment.count().sort_values().plot.bar(
    ylim=0, title= 'Term Frequency of each Emoji \n')
plt.xlabel('\n Number of ocurrences', fontsize = 10);
plt.show()

# پالایش کامنت‌ها
normalizer =  hazm.Normalizer()
tokenizer = hazm.SentenceTokenizer()
tokens = hazm.word_tokenize 
S_Words = list(hazm.stopwords_list())

#بازنمایی متن
tfidf = TfidfVectorizer(lowercase=False, 
                        preprocessor=normalizer.normalize, 
                        tokenizer=tokens,
                        ngram_range=(1, 2),
                        stop_words=S_Words)
comments = df2.comment
features = tfidf.fit_transform(comments).toarray()
labels = df2.category_id

# مقایسه همه مدل‌ها
models = [
    MultinomialNB(),
    RandomForestClassifier(n_estimators=200, max_depth=5, random_state=0),
    LogisticRegression(random_state=0),
    LinearSVC(),
]

CV = 5
cv_df = pd.DataFrame(index=range(CV * len(models)))

entries = []
for model in models:
  model_name = model.__class__.__name__
  accuracies = cross_val_score(model, features, labels, scoring='accuracy', cv=CV)
  for fold_idx, accuracy in enumerate(accuracies):
    entries.append((model_name, fold_idx, accuracy))     
cv_df = pd.DataFrame(entries, columns=['model_name', 'fold_idx', 'accuracy'])

import seaborn as sns
sns.boxplot(x='model_name', y='accuracy', data=cv_df)
sns.stripplot(x='model_name', y='accuracy', data=cv_df, 
              size=8, jitter=True, edgecolor="gray", linewidth=2)
plt.show()

mean_accuracy = cv_df.groupby('model_name').accuracy.mean()
std_accuracy = cv_df.groupby('model_name').accuracy.std()

acc = pd.concat([mean_accuracy, std_accuracy], axis= 1, 
          ignore_index=True)
acc.columns = ['Mean Accuracy', 'Standard deviation']


# یافتن 3 کلمه پرتکرار برای هر ایموجی 
N = 3
for Label, category_id in sorted(category_to_id.items()):
  features_chi2 = chi2(features, labels == category_id)
  indices = np.argsort(features_chi2[0])
  feature_names = np.array(tfidf.get_feature_names())[indices]
  unigrams = [v for v in reversed(feature_names) if len(v.split(' ')) == 1][:N]
  bigrams = [v for v in reversed(feature_names) if len(v.split(' ')) == 2][:N]
  print("# '{}':".format(Label))
  print("  . Top unigrams:\n       . {}".format('\n       . '.join(unigrams)))
  print("  . Top bigrams:\n       . {}".format('\n       . '.join(bigrams)))


# MultinominalNB مدل پیش بینی 
X = df2['comment'] 
y = df2['Label']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)
tfidf_transformer = TfidfTransformer()
count_vect = CountVectorizer()

X_train_counts = count_vect.fit_transform(X_train)
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

clf = MultinomialNB().fit(X_train_tfidf, y_train)
X, y = make_classification(n_samples=2000, n_features=20, n_informative=10, n_classes=11, random_state=0)
estimator = GaussianNB()
y_pred = cross_val_predict(estimator, X, y, cv=10)
print("\n \t\t **************************************")
print("\t\t  MultinominalNB Classification Report  ")
print("\t\t ************************************** \n\n")
print(metrics.classification_report(y, y_pred))
# One example Pridiction ( Without Learning Model !!) .....

exampleComment = " ارزش خرید دارد "
print("Comment: " , exampleComment)
print("Related emoji: ", clf.predict(count_vect.transform([exampleComment])))

# LinearSVC مدل پیش بینی
X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(features, labels, df2.index, test_size=0.8, random_state=1)

model = LinearSVC()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("\n \t\t *********************************")
print("\t\t  LinearSVC Classification Report  ")
print("\t\t ********************************* \n\n")
print(metrics.classification_report(y_test, y_pred, target_names=df['Label'].unique()))

newComment = """عطر بدبوی بود اصلا نخرید"""

X_input = df2['comment']
y_input = df2['Label'] 
X_train, X_test, y_train, y_test = train_test_split(X_input, y_input, 
                                                    test_size=0.8,
                                                    random_state = 0)

fitted_vectorizer = tfidf.fit(X_train)
tfidf_vectorizer_vectors = fitted_vectorizer.transform(X_train)

model = LinearSVC().fit(tfidf_vectorizer_vectors, y_train)

print("Comment: " , newComment)
print("Related emoji: " , model.predict(fitted_vectorizer.transform([newComment])))

# RandomForest مدل پیش بینی 
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size = 0.8, random_state = 21)
                                                                                          
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

Model = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=0)
Model.fit(X_train, y_train)

y_pred = Model.predict(X_test)

print("\n \t\t ************************************")
print("\t\t  RandomForest Classification Report  ")
print("\t\t ************************************ \n\n")
print(metrics.classification_report(y_test, y_pred, target_names= df2['Label'].unique()))
