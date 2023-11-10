
import os
import sys
import json
import subprocess
import shutil
import glob
import zipfile 

import random
os.system("pip install discord.py vaderSentiment scikit-learn pandas numpy transformers")

import discord
from discord.ext import commands
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import os
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import spacy




import numpy as np
import pickle

# Make sure you have installed the required packages:
# pip install discord.py vaderSentiment scikit-learn pandas numpy

# Additional imports for managing the model and data
from pathlib import Path


import torch
from pytorch_pretrained_bert import OpenAIGPTTokenizer, OpenAIGPTModel, OpenAIGPTLMHeadModel






# Create a bot instance
intents = discord.Intents.all()

bot = commands.Bot(command_prefix=commands.when_mentioned_or(','), intents=intents,help_command=None)

# Create a SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()

# Define the folder to store user data
data_folder = 'Users'  # Change the folder name to "Users"

# Ensure the data folder exists
if not os.path.exists(data_folder):
    os.makedirs(data_folder)

# Create a JSON file to store user data or load existing data
data_file = os.path.join(data_folder, 'user_data.json')

if os.path.isfile(data_file):
      with open(data_file, 'r') as file:
          user_data = json.load(file)
else:
      user_data = []

  # Create a Decision Tree Classifier
clf = DecisionTreeClassifier()


# Load a dialogue model and tokenizer (e.g., DialoGPT)


# Create a conversation history for each user
conversation_history = {}

# Define the folder to store user data
data_folder = 'Users'  # Change the folder name to "Users"

# Ensure the data folder exists
if not os.path.exists(data_folder):
    os.makedirs(data_folder)

# Create a JSON file to store user data or load existing data
data_file = os.path.join(data_folder, 'user_data.json')

if os.path.isfile(data_file):
    with open(data_file, 'r') as file:
        user_data = json.load(file)
else:
    user_data = []

@bot.event
async def on_ready():
      print(f'Logged in as {bot.user.name}')



@bot.event
async def on_message(message):
    global user_data, clf

    # Only collect data from user messages, not from the bot itself
    if message.author == bot.user:
        return

    # Analyze the sentiment of the user's message
    sentiment_scores = analyzer.polarity_scores(message.content)

    # Determine the tone based on the sentiment scores
    if sentiment_scores['compound'] >= 0.05:
        tone = "Positive"
    elif sentiment_scores['compound'] <= -0.05:
        tone = "Negative"
    else:
        tone = "Neutral"

    # Store data in the JSON list
    user_data.append({'user_id': message.author.id, 'content': message.content, 'tone': tone})

    # Save the updated JSON data to the file
    with open(data_file, 'w') as file:
        json.dump(user_data, file)

    # Update the decision tree classifier with new data
    accuracy = update_classifier_with_new_data()
    print(f"Classifier accuracy after update: {accuracy}")

    # Check if the user has a conversation history
    if message.author.id not in conversation_history:
        conversation_history[message.author.id] = []

    # Append the user's message to the conversation history
    conversation_history[message.author.id].append(message.content)

    # Generate a response using the decision tree classifier
    input_text = " ".join(conversation_history[message.author.id])
    X_input = vectorizer.transform([input_text])
    tone_prediction = clf.predict(X_input)[0]

    # Generate a response (or handle error) using the GPT model
    try:
        # Load the OpenAI GPT model and tokenizer
        tokenizer_gpt = OpenAIGPTTokenizer.from_pretrained('openai-gpt')
        model_gpt = OpenAIGPTLMHeadModel.from_pretrained('openai-gpt')

        # Continue with the rest of the code for GPT-based response generation
        gpt_input = tokenizer_gpt.encode(input_text, return_tensors="pt")
        gpt_output = model_gpt.generate(gpt_input, max_length=50, num_return_sequences=1, no_repeat_ngram_size=2, top_k=50, top_p=0.95)

        # Extract and send the generated response
        gpt_response = tokenizer_gpt.decode(gpt_output[0], skip_special_tokens=True)
        await message.channel.send(f"GPT Response: {gpt_response}\nTone Prediction: {tone_prediction}")
    except Exception as e:
        await message.channel.send(f"Error in GPT-based response generation: {e}")

# Function to update the decision tree classifier with new data
def update_classifier_with_new_data():
    global user_data, clf, vectorizer

    # Create a bag-of-words representation of the text data
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform([item['content'] for item in user_data])
    y =  np.array([item['tone'] == 'Positive' for item in user_data])

    # Fit the classifier to the data
    clf.fit(X, y)

    # Evaluate the classifier on the entire dataset
    y_pred = clf.predict(X)
    accuracy = accuracy_score(y, y_pred)

    return accuracy

@bot.command()
async def train_model(ctx):
    global user_data_df, clf

    # Send the model training result to the user
    await ctx.send(f"Model retrained.")

@bot.command()
async def save_model(ctx, model_name: str):
    global clf

    # Save the trained model to a file using pickle
    with open(os.path.join(data_folder, f'{model_name}.pkl'), 'wb') as file:
        pickle.dump(clf, file)

    # Send a confirmation message
    await ctx.send(f"Model saved as {model_name}.pkl")

@bot.command()
async def load_model(ctx, model_name: str):
    global clf

    # Load a trained model from a file
    with open(os.path.join(data_folder, f'{model_name}.pkl'), 'rb') as file:
        clf = pickle.load(file)

    # Send a confirmation message
    await ctx.send(f"Model {model_name}.pkl loaded")


# Run the bot with your token
bot.run('MTE2Mjg4MDUzNzQ4MDkzMzU0Nw.G8GT8C.0mDJrMXheydCC7iV83BvLBNqL0_WaBSN9X8qnU')
