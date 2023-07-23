import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Create a TF-IDF vectorizer and Logistic Regression classifier
vectorizer = TfidfVectorizer()
classifier = LogisticRegression()

# Sample training data (you should replace these with your own data)
texts = [
    "Transformers is a deep learning model introduced in the paper 'Attention Is All You Need'.",
    "It has gained popularity for various natural language processing tasks.",
    "BERT is another popular model used for question answering.",
    "It stands for Bidirectional Encoder Representations from Transformers.",
    "The BERT model has been fine-tuned on various tasks including question answering."
]

# Corresponding labels (0 for non-answer, 1 for answer)
labels = [1, 0, 1, 0, 1]

# Transform the text data into numerical features
X = vectorizer.fit_transform(texts)

# Convert labels to NumPy array
y = np.array(labels)

# Train the classifier on the data
classifier.fit(X, y)

#part 2

if __name__ == "__main__":
    # Example questions
    questions = [
        "What is Transformers?",
        "What is the paper 'Attention Is All You Need' about?",
        "For which tasks is BERT commonly used?",
        "What does BERT stand for?",
        "Has the BERT model been fine-tuned on various tasks?"
    ]

    # Transform the questions into numerical features using the same vectorizer
    X_test = vectorizer.transform(questions)

    # Make predictions for the questions
    predictions = classifier.predict(X_test)

    # Print the answers based on the predictions
    for i, question in enumerate(questions):
        answer = "Yes" if predictions[i] == 1 else "No"
        print(f"Question: {question}")
        print(f"Answer: {answer}\n")

