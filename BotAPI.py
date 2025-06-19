import json
import nltk
import numpy as np
import random
from nltk.stem import WordNetLemmatizer
import pickle
from tensorflow.keras.models import load_model
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load  files
lemmatizer = WordNetLemmatizer()
model = load_model('chatbot.h5')
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))

with open("vehicle_advice_Data.json") as data_file:
    intents = json.load(data_file)


# predict response
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


# API route
@app.route('/chatbot', methods=['POST'])
def api_chatbot_response():
    data = request.get_json()
    user_message = data.get("message")

    if not user_message:
        return jsonify({"error": "No message provided"}), 400

    response = chatbot_response(user_message)
    return jsonify({"response": response}), 200


if __name__ == '__main__':
    app.run(debug=True)
