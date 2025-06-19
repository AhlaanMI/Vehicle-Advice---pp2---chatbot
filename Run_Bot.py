import json
import nltk
import numpy as np
import random
from nltk.stem import WordNetLemmatizer
import pickle
from tensorflow.keras.models import load_model

# load
lemmatizer = WordNetLemmatizer()
model = load_model('chatbot.h5')
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
with open("vehicle_advice_Data.json") as data_file:
    intents = json.load(data_file)


def chatbot_response(user_message):
    user_message = nltk.word_tokenize(user_message)
    user_message = [lemmatizer.lemmatize(word.lower()) for word in user_message]
    bag = [0] * len(words)
    for word in user_message:
        if word in words:
            bag[words.index(word)] = 1
    bag = np.array(bag).reshape(1, -1)
    result = model.predict(bag)
    predicted_class = classes[np.argmax(result)]
    for intent in intents['intents']:
        if intent['tag'] == predicted_class:
            responses = intent['responses']
            return random.choice(responses)


def run_chatbot():
    print("Chatbot is running!")
    while True:
        user_message = input("You: ")
        if user_message.lower() == 'exit':
            print("Exiting chatbot...")
            break

        bot_response = chatbot_response(user_message)
        print(f"Bot: {bot_response}")


if __name__ == '__main__':
    run_chatbot()
