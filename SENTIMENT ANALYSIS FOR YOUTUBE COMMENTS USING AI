import os
import pickle
import googleapiclient.discovery
import googleapiclient.errors
import googleapiclient.discovery
import threading
import re
from google.oauth2.credentials import Credentials
from collections import defaultdict
from googleapiclient.discovery import build
from google.auth.transport.requests import Request
from googleapiclient.errors import HttpError
from google_auth_oauthlib.flow import InstalledAppFlow
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('ggplot')

import nltk
!pip install nltk
nltk.download("popular")
nltk.download('vader_lexicon')
import tkinter as tk
from tkinter import messagebox
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from scipy.special import softmax
!pip install transformers
!pip install scipy
MODEL = f"cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)

CLIENT_SECRETS_FILE = "Sentimental analysis\client_secret_287561371646-efo1tjm9p04730h9eb0si5tdf6ec6l46.apps.googleusercontent.com.json"
SCOPES = ['https://www.googleapis.com/auth/youtube.force-ssl']
API_SERVICE_NAME = 'youtube'
API_VERSION = 'v3'
# api_service_name = "youtube"
# api_version = "v3"
DEVELOPER_KEY = "AIzaSyAMyj_Gea0lCersfdXuRUWmGhpptcI6MlI"
# --------------------------------------------Authenticating the user
def get_authenticated_service():
    flow = InstalledAppFlow.from_client_secrets_file(CLIENT_SECRETS_FILE, SCOPES)
    # Use run_local_server instead of run_console
    credentials = flow.run_local_server(port=0)
    return build(API_SERVICE_NAME, API_VERSION, credentials=credentials)

if __name__ == '__main__':
    os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '1'  # Allow insecure transport for local development
    try:
        service = get_authenticated_service()
        print("YouTube API service created successfully.")
    except HttpError as e:
        print(f"An error occurred: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def get_authenticated_service():
    credentials = None
    if os.path.exists('token.pickle'):
        with open('token.pickle', 'rb') as token:
            credentials = pickle.load(token)
    #  Check if the credentials are invalid or do not exist
    if not credentials or not credentials.valid:
        # Check if the credentials have expired
        if credentials and credentials.expired and credentials.refresh_token:
            credentials.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                CLIENT_SECRETS_FILE, SCOPES)
            credentials = flow.run_console()

        # Save the credentials for the next run
        with open('token.pickle', 'wb') as token:
            pickle.dump(credentials, token)
    return build(API_SERVICE_NAME, API_VERSION, credentials = credentials)
  #-----------------------------------------------------------collecting the data
youtube = googleapiclient.discovery.build(API_SERVICE_NAME, API_VERSION, developerKey=DEVELOPER_KEY)
def get_youtube_id():
    """Gets the YouTube video ID from user input."""
    # video_id = youtube_id
    print('Youtube Id = https://www.youtube.com/watch?v= (youtube_id)')
    video_id = input("Enter the Youtube id: ")  # Get the video ID from the user
    return video_id
video_id = get_youtube_id() # Call the function to get the video ID
request = youtube.commentThreads().list(
    part="snippet",
    videoId= video_id,
    maxResults=10000
)
response = request.execute()
#----------------------------------------------------------Store the collected data in the comment comment list
comments = []
for item in response['items']:
    comment = item['snippet']['topLevelComment']['snippet']
    comments.append([
        comment['authorDisplayName'],
        comment['publishedAt'],
        comment['updatedAt'],
        comment['likeCount'],
        comment['textDisplay']
    ])
df = pd.DataFrame(comments, columns=['author', 'published_at', 'updated_at', 'like_count', 'text'])

# print(df.head(1000))
df.to_csv('comments.csv', index=False)

# ------------------------------------------------------------ Cleaning the data
# Function to convert a string to lowercase
def to_lower_case(s):
    if not isinstance(s, str):
        raise ValueError("Input must be a string")
    return s.lower() if s else ""

# Function to remove special characters and numbers
def remove_special_characters(s):
    return re.sub(r'[^a-zA-Z\s]', '', s)

# Function to tokenize a string
def tokenize(s):
    return s.split()

# Function to remove stop words
def remove_stop_words(tokens, stop_words):
    return [word for word in tokens if word not in stop_words]
stop_words = {
        "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves",
        "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them",
        "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am",
        "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did",
        "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by",
        "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above",
        "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further",
        "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few",
        "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too",
        "very", "s", "t", "can", "will", "just", "don", "should", "now"
}
cleaned_comments = []
for item in comments:
    comment_text = item[4]
    # print(f"Original: {comment_text}")
    clean_comment = to_lower_case(comment_text)
    clean_comment = remove_special_characters(clean_comment)
    tokens = tokenize(clean_comment)
    clean_tokens = remove_stop_words(tokens, stop_words)
    final_comment = ' '.join(clean_tokens)
    # print(f"Cleaned: {final_comment}")
    cleaned_comments.append([item[0],final_comment])

# Create a DataFrame from the cleaned comments
df = pd.DataFrame(cleaned_comments, columns=['Author','Cleaned Comment'])


# Save the DataFrame to a CSV file
df.to_csv('cleaned_comments.csv', index=False)

print("Cleaned comments saved to 'cleaned_comments.csv'")

from nltk.sentiment import SentimentIntensityAnalyzer
from tqdm.notebook import tqdm

df = pd.read_csv('cleaned_comments.csv')
df['Cleaned Comment'] = df['Cleaned Comment'].astype(str)
sia = SentimentIntensityAnalyzer()

res = {}
for i, row in df.iterrows():
    comment = row['Cleaned Comment']
    scores = sia.polarity_scores(comment)
    res[i] = scores

df['sentiment_scores'] = pd.Series(res)
for sentiment_type in ['neg', 'neu', 'pos', 'compound']:
    df[sentiment_type] = df['sentiment_scores'].apply(lambda scores: scores.get(sentiment_type, None))
# df = df.drop('sentiment_scores', axis=1)
# df.head(50)

def polarity_scores_roberta(eg):
    encoded_text = tokenizer(eg,return_tensors='pt')
    output = model(**encoded_text) # Fixed indentation
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    sdict = {
        'roberta_neg': scores[0],
        'roberta_neu': scores[1],
        'roberta_pos': scores[2]
    }
    return sdict

# Open the CSV file for writing
import csv
with open('negative_comments.csv', 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    # Write the header row
    writer.writerow(['Author', 'Sentiment', 'Comment'])  # Added 'Comment' column

    for comment_data in cleaned_comments:
        author = comment_data[0]  # Extract author name
        cleaned_comment = comment_data[1]  # Extract cleaned comment

        roberta_scores = polarity_scores_roberta(cleaned_comment)

        if roberta_scores['roberta_neg'] > roberta_scores['roberta_pos']:
            # Write the negative comment data to the CSV file
            writer.writerow([author, 'Negative', cleaned_comment])

print('Negative comments stored with the author name in the "negative_commets.csv"')

# import modin.pandas as pd
df = pd.read_csv('cleaned_comments.csv')
df = df.reset_index() # Call reset_index() to get the DataFrame
df.sort_index(inplace=True)

negative = []
neutral = []
positive = []

for comment in df['Cleaned Comment'] :
  if isinstance(comment, str) and not pd.isna(comment):
    R_score = polarity_scores_roberta(comment)
    negative.append(R_score['roberta_neg'])
    neutral.append(R_score['roberta_neu'])
    positive.append(R_score['roberta_pos'])
  else:
    # Handle invalid comments (e.g., skip or assign default values)
    negative.append(np.nan)  # or any default value
    neutral.append(np.nan)
    positive.append(np.nan)

df['negative'] = negative
df['neutral'] = neutral
df['positive'] = positive
# Assuming scores are already normalized between 0 and 1
df['compound'] = (df['positive'] - df['negative'])
# print(df[['negative','neutral','positive','compound']])
df.to_csv('comments_with_sentiment.csv', index=False)

print("DataFrame with sentiment scores saved to 'comments_with_sentiment.csv'")

!pip install Gradio  # Install Gradio if you haven't already
import gradio as gr
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load your sentiment scores data
df = pd.read_csv('comments_with_sentiment.csv')

# Function to display sentiment scores and plots for a given comment index
def display_sentiment_and_plots(index):
    comment = df.iloc[index]['Cleaned Comment']
    negative = df.iloc[index]['negative']
    neutral = df.iloc[index]['neutral']
    positive = df.iloc[index]['positive']
    compound = df.iloc[index]['compound']

    sentiment_text = f"**Comment:** {comment}\n**Negative:** {negative}\n**Neutral:** {neutral}\n**Positive:** {positive}\n**Compound:** {compound}"

    # Create plots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    sns.barplot(x=df['negative'].value_counts().index, y=df['negative'].value_counts(), ax=axes[0]).set_title('Negative Sentiment')
    sns.barplot(x=df['neutral'].value_counts().index, y=df['neutral'].value_counts(), ax=axes[1]).set_title('Neutral Sentiment')
    sns.barplot(x=df['positive'].value_counts().index, y=df['positive'].value_counts(), ax=axes[2]).set_title('Positive Sentiment')
    plt.tight_layout()


    fig2, axes2 = plt.subplots(figsize=(8, 5))
    sns.histplot(x=df['compound'].value_counts().index, bins=20, ax=axes2).set_title('Compound Sentiment')
    plt.tight_layout()

    return sentiment_text, fig, fig2  # Return text and plots

# Create the Gradio interface
iface = gr.Interface(
    fn=display_sentiment_and_plots,
    inputs=gr.Slider(0, len(df) - 1, step=1, label="Comment Index"),
    outputs=[gr.Textbox(label="Sentiment Analysis Results"), gr.Plot(label="Bar Plots"), gr.Plot(label="Histogram")],
    title="Sentiment Analysis Result",
    description="Select a comment index to view its sentiment scores and plots."
)

iface.launch()
