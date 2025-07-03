# 🤖 AutoMile - Vehicle Advisor Chatbot

![Auto Advisor Chatbot Dashboard](./ccbd631a-51bd-42e5-baed-ae370b2d2284.png)

## 🔍 Overview

**AutoMile** is a **Neural Network-powered chatbot** developed to assist users with vehicle maintenance and repair-related queries. Built as part of my final year research project titled:

> **"IoT-Driven Vehicle Lifecycle Optimization and Maintenance"**

The chatbot uses **Natural Language Processing (NLP)** and a trained deep learning model to provide accurate and real-time automotive advice based on user input.

---

## 🧠 Key Features

- 🔧 Real-Time Car Maintenance Tips & Fixes
- 🗣️ Natural Language Understanding using NLTK
- 📊 Intent Classification with Neural Networks (Keras)
- 🌐 Web Chat Interface with Flask Backend
- 🎙️ Supports Text & Voice Input
- 🛠️ Trained on Custom Vehicle Advice Dataset

---

## 🛠️ Technologies Used

| Component       | Technologies |
|----------------|--------------|
| Frontend       | HTML, CSS, JavaScript |
| Backend        | Python, Flask |
| NLP Tools      | NLTK, WordNet Lemmatizer |
| ML Framework   | TensorFlow, Keras |
| Data Storage   | JSON (for intents), Pickle (for models) |

---

## 📊 Model Architecture

- **Input**: Preprocessed bag-of-words vector
- **Layers**:
  - Dense (128 neurons, ReLU)
  - Dropout (0.5)
  - Dense (64 neurons, ReLU)
  - Dropout (0.5)
  - Output (Softmax over intent classes)
- **Optimizer**: SGD with learning rate decay and momentum
- **Loss**: Categorical Crossentropy
- **Accuracy**: Achieved >95% on training data
- **Epochs**: 100
- **Batch Size**: 5

---

## 📁 Project Structure

├── app.py # Flask app
├── train_model.py # Model training script
├── chatbot.h5 # Trained chatbot model
├── words.pkl # Preprocessed vocabulary
├── classes.pkl # Target class labels
├── vehicle_advice_Data.json # Custom dataset
├── templates/
│ └── index.html # Frontend interface
├── static/ # JS/CSS assets
└── README.md

yaml
Copy
Edit

---

## ⚙️ How to Run Locally

### 🐍 Python Environment
Ensure you have Python 3.x installed.

### 📦 Step 1: Install Dependencies
```bash
pip install -r requirements.txt
If requirements.txt is not provided, install manually:

bash
Copy
Edit
pip install flask tensorflow nltk numpy
▶️ Step 2: Run the Application
bash
Copy
Edit
python app.py
Visit http://127.0.0.1:5000 in your browser to interact with the chatbot.

🚗 Example Questions to Ask the Bot
"How do I take care of my car battery?"

"Why is my engine overheating?"

"What causes brake noise?"

"Can low coolant cause issues?"

The chatbot will respond with helpful automotive guidance based on its trained dataset.

📚 Research Context
This project was developed as part of my final year research module and integrates with a larger IoT-based vehicle monitoring system. The chatbot module is aimed at:

Improving driver knowledge on routine vehicle care

Reducing breakdown incidents through preventive advice

Supporting both fleet operators and individual vehicle owners

📸 User Interface Preview
Auto Advisor Chatbot

👨‍💻 Author
Ahlaan Imamdeen
Final Year Undergraduate | BSc in Software Engineering
📧 ahlaan@example.com
🔗 LinkedIn

📄 License
This project is licensed under the MIT License.

⭐ If you found this project helpful, feel free to star the repo and share feedback!

yaml
Copy
Edit

---

### ✅ Next Steps

If you want, I can:

- Generate a `requirements.txt` file for you.
- Export this as a downloadable `README.md` file.
- Help you write a `LICENSE` file or deploy this to GitHub Pages.

Would you like any of those?
