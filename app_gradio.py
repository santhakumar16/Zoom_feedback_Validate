#Note : create a new src file and put the model and Gradio inside the src then run

import gradio as gr
from transformers import BertTokenizer, BertForSequenceClassification
import torch

# Path to your extracted model directory
#model_path = "./feedback_model"  # Change this!

# Load model and tokenizer
model = BertForSequenceClassification.from_pretrained("./zoom_feedback_validate/feedback_model",local_files_only=True)
tokenizer = BertTokenizer.from_pretrained("./zoom_feedback_validate/feedback_model",local_files_only=True)

# Prediction function
def predict(text, reason):
    inputs = tokenizer(text, reason, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    pred = torch.argmax(outputs.logits, dim=1).item()
    return "Matched" if pred == 1 else "Not Matched."

# Gradio UI
iface = gr.Interface(
    fn=predict,
    inputs=["text", "text"],
    outputs="text",
    title="Feedback Validator",
    description="Check if feedback and reason are logically aligned."
)

iface.launch()
