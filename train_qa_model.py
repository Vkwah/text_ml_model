from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
import numpy as np

def train_online_model(initial_model, vectorizer):
    while True:
        user_input = input("User: ").strip()
        if user_input.lower() == "exit":
            break
        
        X_user = vectorizer.transform([user_input])
        y_user = int(input("Label (0 for negative, 1 for positive): ").strip())
        
        initial_model.partial_fit(X_user, [y_user], classes=[0, 1])

        print("Model has been updated based on your input.")

if __name__ == "__main__":
    # Sample existing dataset (you should replace these with your own data)
    texts = [
        "I love this product!",
        "This is terrible.",
        "Great service!",
        "Awful experience.",
        "Highly recommended.",
        "I would never buy this again.",
    ]

    # Corresponding labels (0 for negative sentiment, 1 for positive sentiment)
    labels = [1, 0, 1, 0, 1, 0]

    # Create a TF-IDF vectorizer and train the initial model
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(texts)
    y = np.array(labels)

    initial_model = SGDClassifier(loss='log', random_state=42)  # Log-loss for probabilistic output
    initial_model.fit(X, y)

    print("Initial Model Training Completed.")

    # Start online training and reply loop
    train_online_model(initial_model, vectorizer)
