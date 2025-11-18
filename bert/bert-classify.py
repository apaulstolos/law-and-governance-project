import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

model_path = "./comment_classifier"
tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)
model = DistilBertForSequenceClassification.from_pretrained(model_path)

model.eval() 

def classify_text(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

    with torch.no_grad():
        outputs = model(**inputs)

    pred = torch.argmax(outputs.logits, dim=1).item()

    label_map = {0: "NS (Non-Substantive)", 1: "S (Substantive)"}
    return label_map[pred]


if __name__ == "__main__":
    while True:
        user_text = input("\nEnter text to classify (or 'quit'): ")
        if user_text.lower() == "quit":
            break
        prediction = classify_text(user_text)
        print("Prediction:", prediction)
