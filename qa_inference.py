from transformers import BertTokenizer, BertForQuestionAnswering
from torch.nn.functional import softmax
import torch

def answer_question(question, context, model, tokenizer, device):
    # ... (same code as in the previous example) ...

if __name__ == "__main__":
    # Example context and question
    context = "Transformers is a deep learning model introduced in the paper 'Attention Is All You Need'. " \
              "It has gained popularity for various natural language processing tasks."
    question = "What is Transformers?"

    # Load the trained model and tokenizer
    model_name = "bert-large-uncased-whole-word-masking-finetuned-squad"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForQuestionAnswering.from_pretrained(model_name)

    # Load the model weights from the saved file
    model_load_path = "qa_model.pt"
    model.load_state_dict(torch.load(model_load_path))

    # Set the device for inference
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Answer the question
    answer = answer_question(question, context, model, tokenizer, device)
    print(f"Question: {question}")
    print(f"Answer: {answer}")
