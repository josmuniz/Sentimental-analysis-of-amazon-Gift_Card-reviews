# -*- coding: utf-8 -*-
"""
Sentiment Anylisis Model Gift Cards 
COMP262 - Group 5
- Juan Arevalo  
- Jose Muniz  
- Ruben Ormeno  
- Samantha Ortiz de Foronda  
- Mariela Ramos Vila

"""

import matplotlib.pyplot as plt
import nltk
import pandas as pd
import seaborn as sns
import scipy.stats as stats
from nltk.corpus import sentiwordnet as swn
from nltk.corpus import wordnet as wn
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.stem import WordNetLemmatizer
from nltk.tag import pos_tag
from sklearn.metrics import f1_score
from nltk.corpus import stopwords
import spacy

"""
1. Dataset data exploration
"""

#Load data
df = pd.read_json(r"C:\MR\Centennial\Sem4\NLP\Project\Gift_Cards_5.json\Gift_Cards_5.json",lines = True)


"""1a. Counts, averages"""
# Base exploration
print('\n---Name and Types of colums---\n')
print(df.info())
print('\n---Shape----\n')
print(df.shape)

# classes by variable
def check_columns(dataframe):
    total_counts = []
    unique_counts = []
    missing_values = []
    for column in dataframe.columns:
        try:
            # Attempt to count unique values in the usual way
            total_count = dataframe[column].count()
            unique_count = dataframe[column].nunique()
            missing_value = dataframe[column].isna().sum()
            
        except TypeError:
            # Handle unhashable items by converting them to strings (or another approach as needed)
            total_count = dataframe[column].astype(str).count()
            unique_count = dataframe[column].astype(str).nunique()
            missing_value = dataframe[column].astype(str).isna().sum() 
                
        total_counts.append(total_count)
        unique_counts.append(unique_count)
        missing_values.append(missing_value)

    # Create DataFrame with counts
    nunique_df = pd.DataFrame({'Column': dataframe.columns, 'Total counts': total_counts,'Unique Value': unique_counts, 'Missing values': missing_values})
    nunique_df = nunique_df.sort_values('Unique Value', ascending=False).reset_index(drop=True)
    return nunique_df
print(check_columns(df))

# Select numerical variables
numerical_variables = df.select_dtypes(include=['int64', 'float64'])

# Print summary statistics
print("\nCounts and Averages of Numerical Variables:")
print(numerical_variables.describe())

# Plotting
plt.figure(figsize=(10, 6))

# Bar plot for counts
plt.subplot(1, 2, 1)
numerical_variables.count().plot(kind='bar')
plt.title('Counts of Numerical Variables')
plt.xlabel('Variables')
plt.ylabel('Count')

# Line plot for averages
plt.subplot(1, 2, 2)
numerical_variables['overall'].value_counts().sort_index().plot(kind='bar', color='skyblue')
plt.title('Distribution of Overall Ratings')
plt.xlabel('Overall Rating')
plt.ylabel('Frequency')
plt.xticks(rotation=0)  # Rotate x-axis labels if needed
plt.grid(axis='y')  

plt.tight_layout()
plt.show()

""" 1b. Distribution of number of reviews across products"""

# Calculate the number of reviews per product (ASIN)
review_counts = df['asin'].value_counts()

# Create the plot
plt.figure(figsize=(10, 6))
review_counts.plot(kind='bar')
plt.title('Distribution of Number of Reviews Across Products')
plt.xlabel('Product ID (ASIN)')
plt.ylabel('Number of Reviews')
plt.xticks([])  # Rotate the x-axis labels for better readability
plt.show()

"""1c. Distribution of number of reviews per products"""


reviews_per_product = df['asin'].value_counts()

# plot using seaborn
plt.figure(figsize=(12, 6))
sns.histplot(reviews_per_product, bins=30, kde=True, color='blue')
plt.title('Distribution of the Number of Reviews Per Product')
plt.xlabel('Number of Reviews')
plt.ylabel('Number of Products')
plt.grid(axis='y', alpha=0.75)

plt.show()



# Assuming 'df' is your DataFrame containing the dataset information

# Calculate the top 10 products with the most reviews and their counts
top_10_most_reviews = df['asin'].value_counts().head(10)

# Calculate the top 10 products with the least reviews and their counts
top_10_least_reviews = df['asin'].value_counts().tail(10)

print("Top 10 products with the most reviews:")
print(top_10_most_reviews)

print("\nTop 10 products with the least reviews:")
print(top_10_least_reviews)

"""1d. Distribution reviews per user"""

# Calculate the number of reviews per user
reviews_per_user = df['reviewerID'].value_counts()

# Plot the distribution of the number of reviews per user using seaborn
plt.figure(figsize=(12, 6))
sns.histplot(reviews_per_user, bins=30, kde=True, color='purple')
plt.title('Distribution of the Number of Reviews Per User')
plt.xlabel('Number of Reviews')
plt.ylabel('Number of Users')
plt.grid(axis='y', alpha=0.75)

plt.show()


# Top 30 users with the most reviews and their counts
top_30_users_reviews = df['reviewerID'].value_counts().head(30)

# Plotting the distribution of reviews for the top 30 users
plt.figure(figsize=(12, 8))
sns.barplot(y=top_30_users_reviews.index, x=top_30_users_reviews.values, palette='coolwarm')
plt.title('Top 30 Users by Number of Reviews')
plt.xlabel('Number of Reviews')
plt.ylabel('User ID')

plt.show()


""" 1e.	Review lengths"""

# Calculate review lengths
df_length = df['reviewText'].dropna().apply(lambda x: len(str(x).split()))

# Plot histogram of review lengths
plt.figure(figsize=(10, 6))
plt.hist(df_length, bins=50, color='skyblue', edgecolor='black')
plt.title('Distribution of Review Lengths')
plt.xlabel('Review Length')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

"""1f.	Analyze lengths"""

# review lengths
print("Minimum review length:", df_length.min())
print("Maximum review length:", df_length.max())
print("Average review length:", df_length.mean())

# Find the mode of review lengths
mode_length = stats.mode(df_length)

print("Mode of review lengths:", mode_length.mode.item())
print("Frequency of the mode:", mode_length.count.item())

""" 1g. Check for duplicates """

duplicates_df = df[df.duplicated(subset=['reviewerID', 'asin', 'reviewTime', 'reviewText'], keep=False)]

# Count the number of duplicate rows based on specific columns
duplicate_count = duplicates_df.shape[0]

print("Number of duplicate rows considering reviewerID, asin, unixReviewTime and reviewText:", duplicate_count)

"""
2.	Text basic pre-processing:
"""

""" 2a. Labeling the data based on ratings """


# Labeling the data based on ratings
def label_rating(row):
    if row['overall'] >= 4:
        return 'positive'
    elif row['overall'] == 3:
        return 'neutral'
    else:  # Ratings 1 and 2
        return 'negative'

df['rating_label'] = df.apply(label_rating, axis=1)

""" 2b. Chose the appropriate columns for your sentiment analyzer """

# Dropping columns 'image', 'vote', and 'style' columns because are not important for the analysis
df.drop(columns=['image', 'vote', 'style'], inplace=True, errors='ignore')


# Dropping 'reviewTime' and 'overall' because 'reviewTime' and 'rating_label' will be used
df.drop(columns=['unixReviewTime', 'overall'], inplace=True)

# Chose the appropriate columns for your sentiment analyzer
df['reviewText'] = df['reviewText'].fillna('')
df['summary'] = df['summary'].fillna('')

# Combine 'reviewText' and 'summary' into a single column for a comprehensive sentiment analysis
df['combined_text'] = df['reviewText'] + " " + df['summary']

df.drop(columns=['reviewText', 'summary'], inplace=True, errors='ignore')

# Print the shape of the DataFrame to see the number of entries and columns
print("DataFrame shape:", df.shape)

print(df.head())


# Print a summary of any missing values in the combined_text column
print("\nMissing values in 'combined_text':", df['combined_text'].isnull().sum())

""" 2c.	Check for outliers """

# Unverified Reviewers outliers

def identify_unverified_reviewers(df):
    unverified_reviewers = df[df['verified'] == False]
    df = df.drop(unverified_reviewers.index)
    return len(unverified_reviewers)


unverified_reviewers_count = identify_unverified_reviewers(df)
print("Number of Unverified Reviewers:", unverified_reviewers_count)


df = df[df['verified'] != False]
print(df.shape)

# Word count outliers

# Check for outliers in 'combined_text' by word count
df['word_count'] = df['combined_text'].fillna('').apply(lambda x: len(x.split()))
print(df['word_count'].describe())


# Calculate z-scores
mean_length = df['word_count'].mean()
std_length = df['word_count'].std()

# Define threshold for outliers using z-scores
z_score_threshold = 3

# Print outlier threshold
outlier_threshold = mean_length + z_score_threshold * std_length
print("Outlier threshold (Z-Score Method):", outlier_threshold)

# Calculate z-scores for word count
wordcount_z_score = (df['word_count'] - mean_length) / std_length

# Identify outliers using z-scores
outliers_z_score = df['word_count'][(wordcount_z_score > z_score_threshold) | (wordcount_z_score < -z_score_threshold)]

# Create a scatter plot to visualize the distribution of review lengths and identify outliers using z-scores
plt.figure(figsize=(8, 6))
plt.scatter(df['word_count'].index, df['word_count'], c='blue', label='Word Count')
plt.scatter(outliers_z_score.index, outliers_z_score, c='red', label='Outliers (Z-Score Method)')
plt.xlabel('Index')
plt.ylabel('Word Count')
plt.title('Distribution of Word Counts with Outliers (Z-Score Method)')
plt.legend()
plt.show()


# Filter rows where word count is less than or equal to 89
df = df[df['word_count'] <= 89]

# Drop the 'word_count' column as it's no longer needed
df.drop(columns=['word_count'], inplace=True)

# Optionally, you can reset the index of the DataFrame after dropping rows
df.reset_index(drop=True, inplace=True)

print(df.shape)

print(df.head())

# Reviewers outliers

# Remove leading and trailing whitespaces and convert to lowercase
df['combined_text'] = df['combined_text'].str.strip().str.lower()

# Group by reviewerID and reviewText_cleaned (cleaned review text) and count the occurrences
review_frequency = df.groupby(['reviewerID', 'combined_text', 'reviewTime', 'rating_label']).size().reset_index(name='review_count')

# Sort the list by review count in descending order
review_frequency_sorted = review_frequency.sort_values(by='review_count', ascending=False)

# Print the sorted DataFrame
print(review_frequency_sorted.head())

# Calculate the mean review count
mean_review_count = review_frequency['review_count'].mean()
print("\nMean review count:", mean_review_count)

# Calculate the standard deviation of review counts
std_review_count = review_frequency['review_count'].std()

# Define a threshold for outliers (3 standard deviations from the mean)
threshold = 3

# Calculate the lower and upper bounds for outliers
lower_bound = mean_review_count - (threshold * std_review_count)
upper_bound = mean_review_count + (threshold * std_review_count)

# Identify outliers based on the bounds
outliers = review_frequency[(review_frequency['review_count'] < lower_bound) | (review_frequency['review_count'] > upper_bound)]

# Print the outliers
print("\nOutliers:")
print(outliers)

# Print the quantity of outliers
print("Quantity of outliers:", len(outliers))

# Store the initial number of rows
initial_rows = len(df)

# Remove duplicate reviews based on reviewerID, cleaned review text, reviewTime, and rating_label
df.drop_duplicates(subset=['reviewerID', 'combined_text', 'reviewTime', 'rating_label'], keep='first', inplace=True)

# Calculate the quantity of deleted rows
deleted_rows = initial_rows - len(df)

# Print the quantity of deleted rows
print("Quantity of deleted rows:", deleted_rows)

# Dropping'reviewerID','asin', 'reviewerName', since they are not directly related to sentiment analysis 
df.drop(columns=['reviewerID','asin', 'reviewerName' ], inplace=True, errors='ignore')

print(df.shape)

# Count the frequency of each rating label
label_counts = df['rating_label'].value_counts()

# Plot the frequency of each rating label using seaborn
plt.figure(figsize=(4, 3))
sns.barplot(x=label_counts.index, y=label_counts.values, palette='magma')
plt.title('Frequency of Rating Labels')
plt.xlabel('Rating Label')
plt.ylabel('Frequency')

plt.show()
print("Shape of dataframe:", df.shape)

df.drop(['verified', 'reviewTime'], axis=1, inplace=True)
print(df.head())


""" # 4. Pre-process the text for VADER """

#  Perform Sentiment Analysis
# Ensure you have the VADER lexicon downloaded
nltk.download('vader_lexicon')

# Initialize the SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()

# Function to get the compound sentiment score for a text
def get_sentiment_score(text):
    return sia.polarity_scores(text)['compound']

# Apply sentiment analysis on the combined review texts
df['sentiment_score'] = df['combined_text'].apply(get_sentiment_score)



df['rating_label'].value_counts()


df['sentiment_score'].describe()


df['rating_label_vader'] = df['sentiment_score'].apply(lambda x: 'positive' if x > 0 
                                                       else ('neutral' 
                                                             #if (x > -0.05 and x < 0.05) 
                                                             if(x == 0)
                                                             else 'negative'))



#Comparing the rating labels and the VADER sentiment labels
# Create a confusion matrix
conf_matrix = pd.crosstab(df['rating_label'], df['rating_label_vader'], rownames=['Actual'], colnames=['Predicted'])

# Plot the confusion matrix using seaborn
plt.figure(figsize=(6,4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix of Rating Labels vs. VADER Sentiment Labels')
plt.show()


#printing the accuracy of the VADER sentiment labels
# Calculate the accuracy of the VADER sentiment labels
accuracy = (conf_matrix['negative']['negative'] + conf_matrix['positive']['positive'] + conf_matrix['neutral']['neutral']) / conf_matrix.values.sum()
print("accuracy of vader lexicon:", accuracy)

F1_score = f1_score(df['rating_label'], df['rating_label_vader'], average='weighted')
print("F1 score of vader lexicon:", F1_score)


#show the reviews that are misclassified
misclassified_reviews = df[df['rating_label'] != df['rating_label_vader']]


print(misclassified_reviews.count())


pd.set_option('display.max_colwidth', None)
misclassified_reviews[(misclassified_reviews['rating_label'] == 'positive') & (misclassified_reviews['rating_label_vader'] == 'negative')][['rating_label','rating_label_vader','combined_text']]


misclassified_reviews[(misclassified_reviews['rating_label']=='positive') & (misclassified_reviews['rating_label_vader']=='negative')][['rating_label','rating_label_vader',]]




""" # 5. Selecting 1000 reviews from the dataset """


#selecting 1000 reviews randomly
df_1000 = df.sample(n=1000, random_state=1)
df_1000.head()



"""  6.A Modeling (Sentiment Analysis) - VADER Lexicon Approach"""


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


df_1000['reviews_after_preprocessing'] = df_1000['combined_text'].apply(preprocess_text)
df_1000['reviews_after_preprocessing'].head()


df_1000['vader_sentiment_score_after_preprocessing'] = df_1000['reviews_after_preprocessing'].apply(get_sentiment_score)
df_1000['rating_label_vader_after_preprocessing'] = df_1000['vader_sentiment_score_after_preprocessing'].apply(lambda x: 'positive' if x > 0
                                                                                                     else ('neutral' if x == 0
                                                                                                           else 'negative'))
confmatrix_after_preprocessing = pd.crosstab(df_1000['rating_label'], df_1000['rating_label_vader_after_preprocessing'], rownames=['Actual'], colnames=['Predicted'])

plt.figure(figsize=(6,4))
sns.heatmap(confmatrix_after_preprocessing, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix of Rating Labels vs. VADER Sentiment Labels After Preprocessing')
plt.show()


""" # 6.B Modeling (Sentiment Analysis) - Sentiwordnet Lexicon Approach"""

# Download necessary NLTK resources
nltk.download('sentiwordnet')
nltk.download('wordnet')

# Function to map PennTreebank tags to WordNet tags
def penn_to_wn(tag):
    if tag.startswith('J'):
        return wn.ADJ
    elif tag.startswith('N'):
        return wn.NOUN
    elif tag.startswith('R'):
        return wn.ADV
    elif tag.startswith('V'):
        return wn.VERB
    return None

# Function to get the SentiWordNet score for a sentence
def get_sentiwordnet_score(text):
    # Tokenize the text and tag the words
    tokens = nltk.word_tokenize(text)
    after_tagging = pos_tag(tokens)

    sentiment = 0.0

    # Initialize the WordNet lemmatizer
    lemmatizer = WordNetLemmatizer()

    # Iterate over the tagged words
    for word, tag in after_tagging:
        wn_tag = penn_to_wn(tag)
        if wn_tag not in (wn.NOUN, wn.ADJ, wn.ADV):
            continue

        # Lemmatize the word
        lemma = lemmatizer.lemmatize(word, pos=wn_tag)
        if not lemma:
            continue

        # Get WordNet synsets for the lemmatized word
        synsets = wn.synsets(lemma, pos=wn_tag)
        if not synsets:
            continue

        # Take the first synset (the most common one)
        synset = synsets[0]
        swn_synset = swn.senti_synset(synset.name())

        # Add the positivity and negativity scores
        sentiment += swn_synset.pos_score() - swn_synset.neg_score()
        # Print
        """print("Token:", word)
        print("Tag:", tag)
        print("Lemma:", lemma)
        print("Synsets:", synsets)
        print("Sentiment:", swn_synset.pos_score() - swn_synset.neg_score())
        print()
        
    # Print text and total sentiment
    print("------------------------------------------")
    print("Texto:", text)
    print("Sentimiento:", sentiment)
    print()
    print("------------------------------------------")"""
    
    return sentiment

df_1000['reviews_after_preprocessing'] = df_1000['combined_text'].apply(get_sentiwordnet_score)
df_1000['reviews_after_preprocessing'].head()


df_1000['rating_label_sentiwordnet_after_preprocessing'] = df_1000['reviews_after_preprocessing'].apply(lambda x: 'positive' if x > 0 else ('neutral' if x == 0 else 'negative'))

confmatrix_after_preprocessing = pd.crosstab(df_1000['rating_label'], df_1000['rating_label_sentiwordnet_after_preprocessing'], rownames=['Actual'], colnames=['Predicted'])

plt.figure(figsize=(6,4))
sns.heatmap(confmatrix_after_preprocessing, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix of Rating Labels vs. SENTIWORDNED Sentiment Labels After Preprocessing')
plt.show()



# Determine misclassified reviews
misclassified_reviews = df_1000[df_1000['rating_label'] != df_1000['rating_label_sentiwordnet_after_preprocessing']]

for index, row in misclassified_reviews.iterrows():
    print("Text:", row['combined_text'])
    print("Actual Label:", row['rating_label'])
    print("Predicted Label:", row['rating_label_sentiwordnet_after_preprocessing'])
    print("=" * 50) 
    
    
#printing the accuracy of the sentiwordnet sentiment labels after preprocessing 
accuracy_after_preprocessing = (confmatrix_after_preprocessing['negative']['negative'] + confmatrix_after_preprocessing['positive']['positive'] + confmatrix_after_preprocessing['neutral']['neutral']) / confmatrix_after_preprocessing.values.sum()
print("accuracy of sentiwordnet lexicon after preprocessing:", accuracy_after_preprocessing)
F1_score_after_preprocessing = f1_score(df_1000['rating_label'], df_1000['rating_label_sentiwordnet_after_preprocessing'], average='weighted')
print("F1 score of sentiwordnet lexicon after preprocessing:", F1_score_after_preprocessing)




print ("TESTING")
test_texts = [
    "This product is amazing!",
    "It is not bad",
    "I hate this product, it's terrible",
    " xxxx", 
    "This product is great!",
    "so so",
    "This product is not bad, but it could be better"
]

# Test VADER model
print("VADER Model:")
for text in test_texts:
    score = sia.polarity_scores(text)['compound']
    if score > 0:
        print(f"Positive: {text}")
    elif score < 0:
        print(f"Negative: {text}")
    else:
        print(f"Neutral: {text}")

# Test SentiWordNet model
print("\nSentiWordNet Model:")
for text in test_texts:
    score = get_sentiwordnet_score(text)
    if score > 0:
        print(f"Positive: {text}")
    elif score < 0:
        print(f"Negative: {text}")
    else:
        print(f"Neutral: {text}")






















