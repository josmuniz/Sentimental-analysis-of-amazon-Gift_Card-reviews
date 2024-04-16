# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 17:11:44 2024

@author: maric
"""

import pandas as pd
import spacy
import random
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from transformers import BartForConditionalGeneration, BartTokenizer



# Load the pre-trained GPT-2 model for summarization
#summarization_pipeline = pipeline("summarization")

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

def identify_unverified_reviewers(df):
    unverified_reviewers = df[df['verified'] == False]
    df = df.drop(unverified_reviewers.index)
    return len(unverified_reviewers)

unverified_reviewers_count = identify_unverified_reviewers(subset_data)

# Delete unverified users reviews 
subset_data = subset_data[subset_data['verified'] != False]
subset_data.shape

# Drop column 'verified'
subset_data.drop(columns=['verified'], inplace=True, errors='ignore')
subset_data.shape

# Find duplicates in the 'reviewText' column
duplicates_combinedText = subset_data[subset_data.duplicated(subset=['reviewText'], keep=False)]

# Count the number of duplicate rows based on specific columns
duplicate_count = duplicates_combinedText.shape[0]

print("Number of duplicate rows considering reviewText:", duplicate_count)

# Remove duplicates from subset_data based on the 'reviewText' column
subset_data = subset_data.drop_duplicates(subset=['reviewText'], keep='first')

subset_data.info()

# Verify the removal of duplicates
print("Number of rows after removing duplicates:", len(subset_data))



"""
Summary
"""
# Select 10 reviews with lengths more than 100 words
selected_reviews = subset_data[subset_data['reviewText'].str.split().apply(len) > 100]
print("Number of reviews with more tha 100 words: ",selected_reviews.shape)

selected_reviews['word_count'] = selected_reviews['reviewText'].str.split().apply(len)
print(selected_reviews[['reviewText', 'word_count']])


# Randomly select 10 reviews from the subset
selected_reviews = selected_reviews.sample(n=10, random_state=42)


# Load BART model and tokenizer
model_name = "facebook/bart-large-cnn"
tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)

# Ensure each review is a string in a list
reviews = selected_reviews['reviewText'].tolist()

# Truncate or pad the input sequences to a maximum length of 512 tokens
encoded_inputs = tokenizer(reviews, padding=True, truncation=True, max_length=512, return_tensors="pt")

# Generate summaries
# Generate summaries
summary_ids = model.generate(encoded_inputs.input_ids, num_beams=4, max_length=50, min_length=40)

# Decode the summaries
summaries = [tokenizer.decode(summary_id, skip_special_tokens=True) for summary_id in summary_ids]

# Print summaries for the first two reviews
for i in range(2):
    print('\n===============================================')
    print(f"Original text for review {i+1}: {reviews[i]}")
    print('\n===============================================')
    print(f"Summary for review {i+1}: {summaries[i]}")



"""
Answering Question
"""    

# Load English language model for spaCy
nlp = spacy.load("en_core_web_sm")

# Function to check if a sentence is a question
def is_question(sentence):
    # Tokenize the sentence
    doc = nlp(sentence)
    
    # Check if the last token is a question mark and the sentence starts with a question word or auxiliary verb
    if len(doc) <= 30 and doc[-1].text == "?" and (doc[0].tag_ == "WDT" or doc[0].tag_ == "WRB" or doc[0].tag_ == "WP" or doc[0].tag_ == "MD"):
        return True
    else:
        return False

# Filter rows containing questions
questions_df = subset_data[subset_data['reviewText'].apply(is_question)]

# Print the questions
print("Questions in the dataset:")
print(questions_df['reviewText'])

# Generate a random integer between 0 and 9 (inclusive)
random_index = random.randint(0, len(questions_df) - 1)
# question_review = questions_df['reviewText'].iloc[random_index]

question_review = questions_df['reviewText'].iloc[random_index]

# Load the language model pipeline
model_name = "gpt2"  
lm_tokenizer = GPT2Tokenizer.from_pretrained(model_name)
lm_model = GPT2LMHeadModel.from_pretrained(model_name)

# Tokenize the input text
inputs = lm_tokenizer.encode(question_review, return_tensors="pt")

# Generate the output
outputs = lm_model.generate(inputs, max_length=100, num_return_sequences=1)

# Decode and print the generated text
generated_response = lm_tokenizer.decode(outputs[0], skip_special_tokens=True)
print('\n===============================================')
print("Original question:", question_review)
print("Generated response:", generated_response)
    

#############################################################################
"""    
# Load English language model for spaCy
nlp = spacy.load("en_core_web_sm")

# Function to check if a sentence is a question
def is_question(sentence):
    # Tokenize the sentence
    doc = nlp(sentence)
    
    # Check if the last token is a question mark and the sentence starts with a question word or auxiliary verb
    if len(doc) <= 30 and doc[-1].text == "?" and (doc[0].tag_ == "WDT" or doc[0].tag_ == "WRB" or doc[0].tag_ == "WP" or doc[0].tag_ == "MD"):
        return True
    else:
        return False

# Filter rows containing questions
questions_df = subset_data[subset_data['reviewText'].apply(is_question)]

# Print the questions
print("Questions in the dataset:")
print(questions_df['reviewText'])

# Generate a random integer between 0 and 9 (inclusive)
random_index = random.randint(0, len(questions_df) - 1)
# question_review = questions_df['reviewText'].iloc[random_index]

question_review = questions_df['reviewText'].iloc[random_index]"""

# Load the language model pipeline

tokenizer = AutoTokenizer.from_pretrained("deepset/roberta-base-squad2")
model = AutoModelForQuestionAnswering.from_pretrained("deepset/roberta-base-squad2")



# Tokenize the input text
inputs = lm_tokenizer.encode(question_review, return_tensors="pt")

# Generate the output
outputs = lm_model.generate(inputs, max_length=100, num_return_sequences=1)

# Decode and print the generated text
generated_response = lm_tokenizer.decode(outputs[0], skip_special_tokens=True)
print('\n===============================================')
print("Original question:", question_review)
print("Generated response:", generated_response)

###################################################################################
    






