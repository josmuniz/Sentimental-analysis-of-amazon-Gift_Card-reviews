# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 06:16:41 2024

Sentiment Anylisis Model Gift Cards
 COMP262 - Group 5
Juan Arevalo  
Jose Muniz  
Ruben Ormeno  
Samantha Ortiz de Foronda  
Mariela Ramos Vila

"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import spacy
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
import itertools
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import mean_squared_error




""" 12. Modeling (Sentiment Analysis) Machine Learning approach """
# Load data
df = pd.read_json(r"C:\MR\Centennial\Sem4\NLP\Project\part b\Gift_Cards.json", lines=True)

# Base exploration
print('\n---Name and Types of colums---\n')
print(df.info())
print('\n---Shape----\n')
print(df.shape)

unique_overall_values = df['overall'].value_counts()
print("Number of unique 'overall' values:", len(unique_overall_values))
print("Counts of each unique 'overall' value:")
print(unique_overall_values)

""" a. Select a subset of the original data minimum 2000 reviews """

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

""" b. Carry out data exploration on the subset and pre-processing
b.1 Data exploration"""


# Verify the balance of classes after manipulation
print(subset_data['overall'].value_counts())

# Base exploration
print('\n---Name and Types of colums---\n')
print(subset_data.info())
print('\n---Shape----\n')
print(subset_data.shape)

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

check_columns(subset_data)

# Select numerical variables
print("\nCounts and Averages of Overall:")
print(subset_data['overall'].describe())

""" Distribution of number of reviews across products """

# Calculate the number of reviews per product (ASIN)
review_counts = subset_data['asin'].value_counts()

# Create the plot
review_counts.plot(kind='bar')
plt.title('Distribution of Number of Reviews Across Products')
plt.xlabel('Product ID (ASIN)')
plt.ylabel('Number of Reviews')
plt.xticks([])  # Rotate the x-axis labels for better readability
plt.show()

review_counts = subset_data['asin'].value_counts()
max_review_count = review_counts.max()

print("Maximum review count:", max_review_count)


"""" Distribution of number of reviews per products """

reviews_per_product = subset_data['asin'].value_counts()

# plot using seaborn
plt.figure(figsize=(12, 6))
sns.histplot(reviews_per_product, bins=30, kde=True, color='blue')
plt.title('Distribution of the Number of Reviews Per Product')
plt.xlabel('Number of Reviews')
plt.ylabel('Number of Products')
plt.grid(axis='y', alpha=0.75)
plt.show()

# Calculate the top 10 products with the most reviews and their counts
top_10_most_reviews = subset_data['asin'].value_counts().head(10)

print("Top 10 products with the most reviews:")
print(top_10_most_reviews)

max_asin = top_10_most_reviews.index[0]  # maximum review count product
max_asin_reviews = subset_data[subset_data['asin'] == max_asin]['reviewText']

print("Reviews for product with ASIN", max_asin, ":", max_asin_reviews)


""" Distribution reviews per user """

# Calculate the number of reviews per user
reviews_per_user = subset_data['reviewerID'].value_counts()

# Calculate the maximum, minimum, and average number of reviews per user
max_reviews_per_user = reviews_per_user.max()
min_reviews_per_user = reviews_per_user.min()
avg_reviews_per_user = reviews_per_user.mean()

print("Maximum number of reviews per user:", max_reviews_per_user)
print("Minimum number of reviews per user:", min_reviews_per_user)
print("Average number of reviews per user:", avg_reviews_per_user)

""" Review lengths """

# Calculate review lengths
df_length = subset_data['reviewText'].dropna().apply(lambda x: len(str(x).split()))

# Plot histogram of review lengths
plt.figure(figsize=(10, 6))
plt.hist(df_length, bins=50, color='skyblue', edgecolor='black')
plt.title('Distribution of Review Lengths')
plt.xlabel('Review Length')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

print("Minimum review length:", df_length.min())
print("Maximum review length:", df_length.max())
print("Average review length:", df_length.mean())



""" b.2 Pre-processing """

# Labeling the data based on ratings
def label_rating(row):
    if row['overall'] >= 4:
        return 'positive'
    elif row['overall'] == 3:
        return 'neutral'
    else:  # Ratings 1 and 2
        return 'negative'

subset_data['rating_label'] = subset_data.apply(label_rating, axis=1)

""" Chose the appropriate columns for your sentiment analyzer """

# Dropping columns 'image', 'vote', and 'style' columns because are not important for the analysis
subset_data.drop(columns=['image', 'vote', 'style'], inplace=True, errors='ignore')


# Dropping 'overall' because'rating_label' will be used
subset_data.drop(columns=['overall'], inplace=True)

# Dropping columns that are not important for Sentiment analysis
subset_data.drop(columns=['reviewerName', 'reviewerID', 'asin', 'unixReviewTime', 'reviewTime' ], inplace=True)

# Chose the appropriate columns for your sentiment analyzer
subset_data['reviewText'] = subset_data['reviewText'].fillna('')
subset_data['summary'] = subset_data['summary'].fillna('')

# Combine 'reviewText' and 'summary' into a single column for a comprehensive sentiment analysis
subset_data['combined_text'] = subset_data['reviewText'] + " " + subset_data['summary']

subset_data.drop(columns=['reviewText', 'summary'], inplace=True, errors='ignore')

# Print the shape of the DataFrame to see the number of entries and columns
print("DataFrame shape:", subset_data.shape)

subset_data.head()
# Print a summary of any missing values in the combined_text column
print("\nMissing values in 'combined_text':", subset_data['combined_text'].isnull().sum())

""" Check for outliers
Unverified Reviewers outliers """

# Identify unverified reviewers
def identify_unverified_reviewers(df):
    unverified_reviewers = df[df['verified'] == False]
    df = df.drop(unverified_reviewers.index)
    return len(unverified_reviewers)

unverified_reviewers_count = identify_unverified_reviewers(subset_data)
print("Number of Unverified Reviewers:", unverified_reviewers_count)

# Delete unverified users reviews 
subset_data = subset_data[subset_data['verified'] != False]

# Drop column 'verified'
subset_data.drop(columns=['verified'], inplace=True, errors='ignore')

""" Word count outliers """

# Check for outliers in 'combined_text' by word count
subset_data['word_count'] = subset_data['combined_text'].fillna('').apply(lambda x: len(x.split()))

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

# Create a scatter plot to visualize the distribution of review lengths and identify outliers using z-scores
plt.figure(figsize=(8, 6))
plt.scatter(subset_data['word_count'].index, subset_data['word_count'], c='blue', label='Word Count')
plt.scatter(outliers_z_score.index, outliers_z_score, c='red', label='Outliers (Z-Score Method)')
plt.xlabel('Index')
plt.ylabel('Word Count')
plt.title('Distribution of Word Counts with Outliers (Z-Score Method)')
plt.legend()
plt.show()

# Filter rows where word count is less than or equal to 89 to delete oouliers
subset_data = subset_data[subset_data['word_count'] <= outlier_threshold]

# Drop the 'word_count' column as it's no longer needed
subset_data.drop(columns=['word_count'], inplace=True)

# Optionally, you can reset the index of the DataFrame after dropping rows
subset_data.reset_index(drop=True, inplace=True)

subset_data.shape


""" Duplicates """
# Find duplicates in the 'combined_text' column
duplicates_combinedText = subset_data[subset_data.duplicated(subset=['combined_text'], keep=False)]

# Count the number of duplicate rows based on specific columns
duplicate_count = duplicates_combinedText.shape[0]

print("Number of duplicate rows considering combined_text:", duplicate_count)

# Remove duplicates from subset_data based on the 'combined_text' column
subset_data = subset_data.drop_duplicates(subset=['combined_text'], keep='first')

# Verify the removal of duplicates
print("Number of rows after removing duplicates:", len(subset_data))

# Count the frequency of each rating label
label_counts = subset_data['rating_label'].value_counts()

# Calculate percentage of each rating label
label_percentage = label_counts / len(subset_data) * 100

# Plot the frequency of each rating label using seaborn
plt.figure(figsize=(4, 3))
sns.barplot(x=label_counts.index, y=label_counts.values, palette='magma')
plt.title('Frequency of Rating Labels')
plt.xlabel('Rating Label')
plt.ylabel('Frequency')

plt.show()


# Print the percentage for each rating label
for label, percentage in zip(label_percentage.index, label_percentage.values):
    print(f"{label}: {percentage:.2f}%")
# Print shape of dataframe
print("Shape of dataframe:", subset_data.shape)

""" Pre processing """

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

subset_data['reviews_after_preprocessing'] = subset_data['combined_text'].apply(preprocess_text)
subset_data['reviews_after_preprocessing'].head()


from sklearn.model_selection import StratifiedShuffleSplit

X = subset_data['reviews_after_preprocessing'].values
y = subset_data['rating_label'].values


# Initialize StratifiedShuffleSplit
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)

# Generate indices for splitting
for train_index, test_index in sss.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)


""" d. Represent the text using one of the text represtations discussed in the course, 
make sure to note in your report why you chose that representation. """

# Function to load GloVe embeddings from file
def load_glove_embeddings(file_path):
    embeddings_index = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    return embeddings_index

# Load GloVe embeddings (example file path)
glove_file_path = r"C:\MR\Centennial\Sem4\NLP\Lab 6\glove.6B\glove.6B.100d.txt" 
glove_embeddings = load_glove_embeddings(glove_file_path)

# Function to vectorize text using GloVe embeddings
def vectorize_text_with_glove(text, embeddings_index, embedding_dim):
    vectorized_text = []
    for sentence in text:
        sentence_vector = []
        for word in sentence.split():
            if word in embeddings_index:
                sentence_vector.append(embeddings_index[word])
        if sentence_vector:
            vectorized_text.append(np.mean(sentence_vector, axis=0))
        else:
            vectorized_text.append(np.zeros(embedding_dim))
    return np.array(vectorized_text)

embedding_dim = 100  # Assuming GloVe embeddings of dimension 100
X_train_textr = vectorize_text_with_glove(X_train, glove_embeddings, embedding_dim)
X_test_textr = vectorize_text_with_glove(X_test, glove_embeddings, embedding_dim)



""" e. Build two sentiment analysis models using 70% of the data. Choose two of the following 
Machine Learning algorithms to build and fine tune your models:"""

""" Logistic Regression """
# Build and fine-tune sentiment analysis models
# Model 1: Logistic Regression
logreg = LogisticRegression(class_weight='balanced') #instantiate a logistic regression model
logreg.fit(X_train_textr, y_train) #fit the model with training data

#Make predictions on test data
y_pred_class = logreg.predict(X_test_textr)


#Step 4: Evaluate the classifier using various measures

# Function to plot confusion matrix. 


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label',fontsize=15)
    plt.xlabel('Predicted label',fontsize=15)
    
y_pred_prob = logreg.predict_proba(X_test_textr)

acc_lg_test = accuracy_score(y_test, y_pred_class)
#calculate evaluation measuresfor Logistic Regression
print("Accuracy: ", acc_lg_test)
print("AUC: ", roc_auc_score(y_test, y_pred_prob, multi_class='ovr'))
# Other metrics
print("Classification Report:")
print(classification_report(y_test, y_pred_class))
cnf_matrix = confusion_matrix(y_test, y_pred_class)
plt.figure(figsize=(8,6))

plot_confusion_matrix(cnf_matrix, classes=['Negative','Neutral', 'Positive'],normalize=True,
                      title='Confusion matrix with all features')

""" Support Vector Machine (SVM) """


# Instantiate the SVM model with 'rbf' kernel and class_weight='balanced'
svm = SVC(kernel='rbf', class_weight='balanced', probability=True)

# Fit the model with training data
svm.fit(X_train_textr, y_train)

# Make predictions on test data
y_pred_class = svm.predict(X_test_textr)

# Obtain predicted probabilities for each class
y_pred_prob = svm.predict_proba(X_test_textr)

acc_SVM_test = accuracy_score(y_test, y_pred_class)
# Calculate evaluation measures:
print("Accuracy: ", acc_SVM_test)
print("AUC: ", roc_auc_score(y_test, y_pred_prob, multi_class='ovr'))

# Other metrics
print("Classification Report:")
print(classification_report(y_test, y_pred_class))

# Plot confusion matrix
cnf_matrix = confusion_matrix(y_test, y_pred_class)
plt.figure(figsize=(8,6))
plot_confusion_matrix(cnf_matrix, classes=['Negative','Neutral', 'Positive'],normalize=True,
                      title='Confusion matrix with all features')

""" Naive Bayes: """
# Initialize Gaussian Naive Bayes classifier
gaussian_nb = GaussianNB()

# Train the model on the training data
gaussian_nb.fit(X_train_textr, y_train)

# Predict sentiment labels for the testing data
y_pred_class = gaussian_nb.predict(X_test_textr)

# Obtain predicted probabilities for each class
y_pred_prob = gaussian_nb.predict_proba(X_test_textr)

acc_NB_test = accuracy_score(y_test, y_pred_class)
#calculate evaluation measures:
print("Accuracy: ", acc_NB_test)
print("AUC: ", roc_auc_score(y_test, y_pred_prob, multi_class='ovr'))
# Other metrics
print("Classification Report:")
print(classification_report(y_test, y_pred_class))
cnf_matrix = confusion_matrix(y_test, y_pred_class)
plt.figure(figsize=(8,6))

plot_confusion_matrix(cnf_matrix, classes=['Negative','Neutral', 'Positive'],normalize=True,
                      title='Confusion matrix with all features')

""" Gradient Boosting: """

# Initialize the Gradient Boosting model
gradient_boosting = GradientBoostingClassifier()

# Train the model on the training data
gradient_boosting.fit(X_train_textr, y_train)

# Predict sentiment labels for the testing data
y_pred_class = gradient_boosting.predict(X_test_textr)

# Obtain predicted probabilities for each class
y_pred_prob = gradient_boosting.predict_proba(X_test_textr)

acc_GB_test = accuracy_score(y_test, y_pred_class)
#calculate evaluation measures:
print("Accuracy: ", acc_GB_test)
print("AUC: ", roc_auc_score(y_test, y_pred_prob, multi_class='ovr'))
# Other metrics
print("Classification Report:")
print(classification_report(y_test, y_pred_class))
cnf_matrix = confusion_matrix(y_test, y_pred_class)
plt.figure(figsize=(8,6))

plot_confusion_matrix(cnf_matrix, classes=['Negative','Neutral', 'Positive'],normalize=True,
                      title='Confusion matrix with all features')

""" Multi-layer Perceptron (MLP): """

# Initialize the MLP model
mlp_model = MLPClassifier()

# Train the model on the training data
mlp_model.fit(X_train_textr, y_train)

# Predict sentiment labels for the testing data
y_pred_class = mlp_model.predict(X_test_textr)

# Obtain predicted probabilities for each class
y_pred_prob = mlp_model.predict_proba(X_test_textr)

acc_MLP_test = accuracy_score(y_test, y_pred_class)

#calculate evaluation measures:
print("Accuracy: ", acc_MLP_test)
print("AUC: ", roc_auc_score(y_test, y_pred_prob, multi_class='ovr'))
# Other metrics
print("Classification Report:")
print(classification_report(y_test, y_pred_class))
cnf_matrix = confusion_matrix(y_test, y_pred_class)
plt.figure(figsize=(8,6))

plot_confusion_matrix(cnf_matrix, classes=['Negative','Neutral', 'Positive'],normalize=True,
                      title='Confusion matrix with all features')


print('\n=========================================')
print('ACCURACY OF MODELS WITH DATASET TESTING')
print("Accuracy Logistic Regresion: {:.2%}".format(acc_lg_test))
print("Accuracy SVM Model: {:.2%}".format(acc_SVM_test))
print("Accuracy Naive Bayes Model: {:.2%}".format(acc_NB_test))
print("Accuracy Gradient Boosting: {:.2%}".format(acc_GB_test))
print("Accuracy MLP: {:.2%}".format(acc_MLP_test))





# Testing the models with some texts
print("TESTING WITH ADDITIONALS REVIWES")

# Test texts
test_texts = [
    "This product is amazing!",
    "It is not bad",
    "I hate this product, it's terrible",
    " xxxx", 
    "This product is great!",
    "so so",
    "This product is not bad, but it could be better"
]

y_texts = [   
    "positive",
    "neutral",
    "negative",
    "neutral", 
    "positive",
    "neutral",
    "neutral"
    ]
# Preprocess the test texts
preprocessed_test_texts = [preprocess_text(text) for text in test_texts]

# Vectorize the preprocessed test texts using GloVe embeddings
X_test_vectorized = vectorize_text_with_glove(preprocessed_test_texts, glove_embeddings, embedding_dim)

# Test Logistic Regression model
print("Logistic Regression Model:")
# Make predictions
y_pred_logreg = logreg.predict(X_test_vectorized)
print("Predictions:", y_pred_logreg)
# Calculate accuracy
accuracy_logreg = accuracy_score(y_texts, y_pred_logreg)
print("Accuracy Logistic Regresion:", accuracy_logreg)

# Test SVM model
print("\nSVM Model:")
# Make predictions
y_pred_svm = svm.predict(X_test_vectorized)
print("Predictions:", y_pred_svm)
# Calculate accuracy
accuracy_svm = accuracy_score(y_texts, y_pred_svm)
print("Accuracy SVM Model:", accuracy_svm)

# Test Naive Bayes model
print("\nNaive Bayes Model:")
# Make predictions
y_pred_naive_bayes = gaussian_nb.predict(X_test_vectorized)
print("Predictions:", y_pred_naive_bayes)
# Calculate accuracy
accuracy_naive_bayes = accuracy_score(y_texts, y_pred_naive_bayes)
print("Accuracy Naive Bayes Model:", accuracy_naive_bayes)

# Test Gradient Boosting model
print("\nGradient Boosting Model:")
# Make predictions
y_pred_gradient_boosting = gradient_boosting.predict(X_test_vectorized)
print("Predictions:", y_pred_gradient_boosting)
# Calculate accuracy
accuracy_gradient_boosting = accuracy_score(y_texts, y_pred_gradient_boosting)
print("Accuracy Gradient Boosting:", accuracy_gradient_boosting)

# Test MLP model
print("\nMLP Model:")
# Make predictions
y_pred_mlp = mlp_model.predict(X_test_vectorized)
print("Predictions:", y_pred_mlp)
# Calculate accuracy
accuracy_mlp = accuracy_score(y_texts, y_pred_mlp)
print("Accuracy MLP:", accuracy_mlp)


import pickle
# Load the VADER model from the file
with open(r"C:\MR\Centennial\Sem4\NLP\Project\part b\models\vader_model_part1.pkl", 'rb') as f:
    vader_model = pickle.load(f)
    
    


# Test VADER model
print("VADER Model:")
y_pred_vader = []
for text in preprocessed_test_texts:
    score = vader_model.polarity_scores(text)['compound']
    if score > 0:
        print("positive")
        y_pred_vader.append('positive')
    elif score < 0:
        print("negative")
        y_pred_vader.append('negative')
    else:
        print("neutral")
        y_pred_vader.append('neutral')


# Calculate accuracy
accuracy_vader = accuracy_score(y_texts, y_pred_vader)
print("Accuracy VADER:", accuracy_vader)

print('\n=========================================')
print("Accuracy Logistic Regresion: {:.2%}".format(accuracy_logreg))
print("Accuracy SVM Model: {:.2%}".format(accuracy_svm))
print("Accuracy Naive Bayes Model: {:.2%}".format(accuracy_naive_bayes))
print("Accuracy Gradient Boosting: {:.2%}".format(accuracy_gradient_boosting))
print("Accuracy MLP: {:.2%}".format(accuracy_mlp))
print("Accuracy VADER: {:.2%}".format(accuracy_vader))



""" 16. a.Enhance the rating values """


def enhanced_rating_predictions(X_test, y_test):
    inferred_lg = logreg.predict(X_test)
    inferred_svm = svm.predict(X_test)
    
    print(type(inferred_lg))
    print(type(inferred_svm))
    
    best_weights = None
    lowest_mse = float('inf')
    
    # Grid search for weights
    for alpha in range(0, 11):
        lg_weight = alpha / 10
        svm_weight = 1 - lg_weight
        
        # Combine predictions using a linear combination
        enhanced_rating = np.multiply(lg_weight, inferred_lg) + np.multiply(svm_weight, 1 - inferred_svm)
        
        # Calculate MSE
        mse = mean_squared_error(y_test, enhanced_rating)
        
        # Check if this MSE is the lowest found so far
        if mse < lowest_mse:
            lowest_mse = mse
            best_weights = (lg_weight, svm_weight)
    
    print("Best Weights:", best_weights)
    
    # Recalculate enhanced_rating using the best weights
    enhanced_rating = np.multiply(best_weights[0], inferred_lg) + np.multiply(best_weights[1], 1 - inferred_svm)
    
    mse = mean_squared_error(y_test, enhanced_rating)
    print("MSE on the testing data:", mse)
    
    # Create DataFrame to store final results
    final_results = pd.DataFrame()
    final_results["actual_rating"] = y_test
    final_results["enhanced_rating"] = enhanced_rating.round().astype(int)
    
    # Print classification report
    print(classification_report(final_results["actual_rating"], final_results["enhanced_rating"]))
    
    return final_results

final_test_results = enhanced_rating_predictions(X_test_textr, y_test)



print("Records where enhanced rating differs from actual rating:")
for index in final_test_results.index:
    if final_test_results.at[index, "actual_rating"] != final_test_results.at[index, "enhanced_rating"]:
        #print("Review Text:", df_final[index])
        print("Actual Rating:", final_test_results.at[index, "actual_rating"])
        print("Enhanced Rating:", final_test_results.at[index, "enhanced_rating"])




























