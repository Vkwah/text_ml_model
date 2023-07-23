from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
import numpy as np

# Import the train_online_model function from 'train_online_model.py'
from train_qa_model import train_online_model

def reply_sentiment(user_input, model, vectorizer):
    X_user = vectorizer.transform([user_input])
    prediction = model.predict(X_user)

    return "Positive" if prediction[0] == 1 else "Negative"

if __name__ == "__main__":
    # Load the trained model from 'train_online_model.py' and vectorizer
    # For this example, we will reuse the same dataset and model.
    # In a real-world scenario, you can save/load the model and vectorizer using pickle or other methods.

    texts = [
        "I love this product!",
        "This is terrible.",
        "Great service!",
        "Awful experience.",
        "Highly recommended.",
        "I would never buy this again.",
    ]

    labels = [1, 0, 1, 0, 1, 0]

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(texts)
    y = np.array(labels)

    initial_model = SGDClassifier(loss='hinge', random_state=42)
    initial_model.fit(X, y)

    print("Initial Model Loaded.")

    # Example usage of the trained model to reply based on user input
    while True:
        user_input = input("User: ").strip()
        if user_input.lower() == "exit":
            break
        
        reply = reply_sentiment(user_input, initial_model, vectorizer)
        print(f"Bot: Sentiment is {reply}.")
