# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 15:19:36 2024

@author: maric
"""



import pandas as pd
import spacy
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import numpy as np


#Load data
df = pd.read_json(r"C:\MR\Centennial\Sem4\NLP\Project\part b\Gift_Cards.json",lines = True)

# Select a Subset of Data
subset_data = df

# Subsampling of each rating category
subset_1 = subset_data[subset_data['overall'] == 1].sample(n=1400, replace=True, random_state=42)
subset_2 = subset_data[subset_data['overall'] == 2].sample(n=1400, replace=True, random_state=42)
subset_3 = subset_data[subset_data['overall'] == 3].sample(n=1400, replace=True, random_state=42)
subset_4 = subset_data[subset_data['overall'] == 4].sample(n=1400, replace=True, random_state=42)
subset_5 = subset_data[subset_data['overall'] == 5].sample(n=1400, replace=True, random_state=42)

# Combine the subsampled samples
subset_data = pd.concat([subset_1, subset_2, subset_3, subset_4, subset_5])


# Labeling the data based on ratings
def label_rating(row):
    if row['overall'] >= 4:
        return 'positive'
    elif row['overall'] == 3:
        return 'neutral'
    else:  # Ratings 1 and 2
        return 'negative'

subset_data['rating_label'] = subset_data.apply(label_rating, axis=1)

# Dropping columns 'image', 'vote', and 'style' columns because are not important for the analysis
subset_data.drop(columns=['image', 'vote', 'style'], inplace=True, errors='ignore')

# Dropping 'overall' because'rating_label' will be used
subset_data.drop(columns=['overall'], inplace=True)

# Dropping columns that are not important for Sentiment analysis
subset_data.drop(columns=['reviewerName', 'reviewerID', 'asin', 'unixReviewTime', 'reviewTime' ], inplace=True)

# Chose the appropriate columns for your sentiment analyzer
subset_data['reviewText'] = subset_data['reviewText'].fillna('')

subset_data.drop(columns=['summary'], inplace=True, errors='ignore')

# Print the shape of the DataFrame to see the number of entries and columns
print("DataFrame shape:", subset_data.shape)

# Print a summary of any missing values in the reviewText column
print("\nMissing values in 'reviewText':", subset_data['reviewText'].isnull().sum())

def identify_unverified_reviewers(df):
    unverified_reviewers = df[df['verified'] == False]
    df = df.drop(unverified_reviewers.index)
    return len(unverified_reviewers)

unverified_reviewers_count = identify_unverified_reviewers(subset_data)
print("Number of Unverified Reviewers:", unverified_reviewers_count)

# Delete unverified users reviews 
subset_data = subset_data[subset_data['verified'] != False]
subset_data.shape

# Drop column 'verified'
subset_data.drop(columns=['verified'], inplace=True, errors='ignore')
subset_data.shape

# Check for outliers in 'reviewText' by word count
subset_data['word_count'] = subset_data['reviewText'].fillna('').apply(lambda x: len(x.split()))

# Calculate z-scores
mean_length = subset_data['word_count'].mean()
std_length = subset_data['word_count'].std()

# Define threshold for outliers using z-scores
z_score_threshold = 3

# Print outlier threshold
outlier_threshold = mean_length + z_score_threshold * std_length
print("Outlier threshold (Z-Score Method):", outlier_threshold)

# Calculate z-scores for word count
wordcount_z_score = (subset_data['word_count'] - mean_length) / std_length

# Identify outliers using z-scores
outliers_z_score = subset_data['word_count'][(wordcount_z_score > z_score_threshold) | (wordcount_z_score < -z_score_threshold)]


# Filter rows where word count is less than or equal to 89 to delete oouliers
subset_data = subset_data[subset_data['word_count'] <= outlier_threshold]

# Drop the 'word_count' column as it's no longer needed
subset_data.drop(columns=['word_count'], inplace=True)

# Optionally, you can reset the index of the DataFrame after dropping rows
subset_data.reset_index(drop=True, inplace=True)

subset_data.shape

# Find duplicates in the 'reviewText' column
duplicates_combinedText = subset_data[subset_data.duplicated(subset=['reviewText'], keep=False)]

# Count the number of duplicate rows based on specific columns
duplicate_count = duplicates_combinedText.shape[0]

print("Number of duplicate rows considering reviewText:", duplicate_count)

# Remove duplicates from subset_data based on the 'reviewText' column
subset_data = subset_data.drop_duplicates(subset=['reviewText'], keep='first')

# Verify the removal of duplicates
print("Number of rows after removing duplicates:", len(subset_data))

# Count the frequency of each rating label
label_counts = subset_data['rating_label'].value_counts()

# Calculate percentage of each rating label
label_percentage = label_counts / len(subset_data) * 100


# Print the percentage for each rating label
for label, percentage in zip(label_percentage.index, label_percentage.values):
    print(f"{label}: {percentage:.2f}%")
# Print shape of dataframe
print("Shape of dataframe:", subset_data.shape)


# Keep only rows with positive and neutral ratings after removing duplicates based on 'reviewText'
subset_data_positive = subset_data[(subset_data['rating_label'] == 'positive') | (subset_data['rating_label'] == 'neutral')]


subset_data.shape



# Load spaCy model for tokenization
nlp = spacy.load('en_core_web_sm')

# Define the list of stopwords and create the lemmatizer object
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    # Tokenize the text using spaCy
    doc = nlp(text)
    tokens = [token.text for token in doc]

    # Remove stopwords
    tokens = [token for token in tokens if token.lower() not in stop_words]

    # Lemmatize each word
    tokens = [lemmatizer.lemmatize(token) for token in tokens]

    # Join tokens back into text
    preprocessed_text = ' '.join(tokens)

    return preprocessed_text


subset_data['reviewText'] = subset_data['reviewText'].apply(preprocess_text)
subset_data['reviewText'].head()


"""
class GiftCardRecommender:
    def __init__(self, data):
        self.data = data
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.knn_model = NearestNeighbors(metric='cosine', algorithm='brute')

    def preprocess_data(self):
        # TF-IDF vectorization
        self.tfidf_matrix = self.vectorizer.fit_transform(self.data['reviewText'])

    def fit_knn_model(self):
        # Fitting the kNN model
        self.knn_model.fit(self.tfidf_matrix)

    def recommend_gift_cards(self, query_text, top_n=5):
        # Transform query text into TF-IDF vector
        query_vector = self.vectorizer.transform([query_text])
        
        # Find k nearest neighbors
        distances, indices = self.knn_model.kneighbors(query_vector, n_neighbors=top_n)
        
        # Return recommended gift cards
        return self.data.iloc[indices[0]]['reviewText']import numpy as np
"""
class GiftCardRecommender:
    def __init__(self, data):
        self.data = data
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.knn_model = NearestNeighbors(metric='cosine', algorithm='brute')

    def preprocess_data(self):
        # TF-IDF vectorization
        self.tfidf_matrix = self.vectorizer.fit_transform(self.data['reviewText'])

    def fit_knn_model(self):
        # Fitting the kNN model
        self.knn_model.fit(self.tfidf_matrix)

    def recommend_gift_cards(self, query_text, user_context=None, top_n=5, alpha=0.5):
        # Transform query text into TF-IDF vector
        query_vector = self.vectorizer.transform([query_text])
        
        # Find k nearest neighbors
        distances, indices = self.knn_model.kneighbors(query_vector, n_neighbors=top_n)
        
        # Calculate context score for each recommended item
        context_scores = []
        if user_context is None:
            # Default context 
            user_context = {'general': True}

        for index in indices[0]:
            # Assuming 'context' is a column in your data
            item_context = self.data.iloc[index].get('context', {'general': True})
            context_similarity = self.compute_context_similarity(user_context, item_context)
            context_scores.append(context_similarity)
        
        # Calculate utility scores for recommended items
        utility_scores = (1 - alpha) * np.array(context_scores) + alpha * distances.flatten()
        
        # Sort the recommendations by utility score
        sorted_indices = np.argsort(utility_scores)
        sorted_recommendations = self.data.iloc[indices[0][sorted_indices]]
        
        return sorted_recommendations

    def compute_context_similarity(self, user_context, item_context):
        # Placeholder for context similarity calculation
        # You can implement your context similarity calculation here
        return 0.5  # Placeholder, replace with your actual calculation




"""
Recommender function
"""
"""
def main():
    # Initialize the recommender system
    recommender = GiftCardRecommender(subset_data)
    # Preprocess the data
    recommender.preprocess_data()
    # Fit the kNN model
    recommender.fit_knn_model()
    # Interaction loop
    while True:
        user_input = input("Enter a song title (or 'exit' to quit): ")
        user_input = preprocess_text(user_input)
        if user_input == 'exit':
            print("Exiting...")
            break
        else:
            # Example query text
            query_text = user_input

            # Get recommendations
            recommendations = recommender.recommend_gift_cards(query_text)
            print("Top 5 recommended gift cards:")
            print(recommendations)
            

#        else:
#            print(f"We don't have recommendations for '{user_input}'")

# Execute main function
if __name__ == "__main__":
    main()

"""

def main():
    # Initialize the recommender system
    recommender = GiftCardRecommender(subset_data)
    # Preprocess the data
    recommender.preprocess_data()
    # Fit the kNN model
    recommender.fit_knn_model()
    # Interaction loop
    while True:
        user_input = input("Enter a query (or 'exit' to quit): ")
        if user_input.lower() == 'exit':
            print("Exiting...")
            break
        else:
            # Example query text
            query_text = preprocess_text(user_input)

            # Default user context (you can adjust this as needed)
            user_context = {'general': True}

            # Get recommendations
            recommendations = recommender.recommend_gift_cards(query_text, user_context)
            print("Top 5 recommended gift cards:")
            print(recommendations)

# Execute main function
if __name__ == "__main__":
    main()