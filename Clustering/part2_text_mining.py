# Import necessary libraries
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import pandas as pd

corpus = [
    'Now for manners use has company believe parlors.',
    'Least nor party who wrote while did. Excuse formed as is agreed admire so on result parish.',
    'Put use set uncommonly announcing and travelling. Allowance sweetness direction to as necessary.',
    'Principle oh explained excellent do my suspected conveying in.',
    'Excellent you did therefore perfectly supposing described.',
    'Its had resolving otherwise she contented therefore.',
    'Afford relied warmth out sir hearts sister use garden.',
    'Men day warmth formed admire former simple.',
    'Humanity declared vicinity continue supplied no an. He hastened am no property exercise of.',
    'Dissimilar comparison no terminated devonshire no literature on. Say most yet head room such just easy.'
]

# Step 1: Count Vectorization
count_vectorizer = CountVectorizer()
count_vector = count_vectorizer.fit_transform(corpus)
count_df = pd.DataFrame(count_vector.toarray(), columns=count_vectorizer.get_feature_names_out())

# Step 2: TF-IDF Vectorization
tfidf_vectorizer = TfidfVectorizer()
tfidf_vector = tfidf_vectorizer.fit_transform(corpus)
tfidf_df = pd.DataFrame(tfidf_vector.toarray(), columns=tfidf_vectorizer.get_feature_names_out())

# Step 3: Display the results
print("Count Vector:\n", count_df)
print("\nTF-IDF Vector:\n", tfidf_df)

# Step 4: Strip and Compress
tfidf_vectorizer = TfidfVectorizer(
    stop_words='english', # Removes common English stop words like "and", "the", etc.
    max_features=1000,    # Limits the vectorizer to the top 1000 most frequent terms across the corpus.
    min_df=1              # A term must appear in at least 1 document to be included.
)

tfidf_vector = tfidf_vectorizer.fit_transform(corpus)
print(type(tfidf_vector))
print(tfidf_vector.toarray())
print(tfidf_vectorizer.get_feature_names_out())

