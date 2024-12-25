# Automated Document Categorization and Summarization

This repository provides a solution for automating the categorization and summarization of documents at Appian Credit Union. The project leverages state-of-the-art OCR, machine learning, and deep learning techniques to process thousands of daily incoming PDFs and images efficiently.

# Problem Statement

## Appian Credit Union processes:

Applications for bank accounts (credit cards, savings accounts)

Identity documents (driver’s licenses, passports)

Supporting financial documents (income statements, tax returns)

Receipts

Manually verifying and organizing these documents is time-consuming and repetitive. This project automates the process to:

Associate documents with the correct individual.

Categorize documents by type.

Generate concise summaries for quick review.

## Pipeline Overview

1.Data Preprocessing

2.Text Extraction

3.Document Categorization

4.Summarization

5.Evaluation and Optimization

## Technologies and Tools

OpenCV: Image preprocessing.

Tesseract OCR: Text extraction.

Transformers (Hugging Face): NLP models for classification and summarization.

PyTorch: Deep learning framework.

Pandas: Data manipulation.

Scikit-learn: Evaluation and hyperparameter tuning.

## Steps and Code Snippets

### 1. Data Preprocessing

Preprocess the images and PDFs to improve OCR accuracy.

import cv2
<pre>
<strong style="background-color:#2d2d2d; color:#ffffff; padding: 8px; border-radius: 6px;">📄 main.py</strong>
<code>
def preprocess_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)[1]
    image = cv2.medianBlur(image, 3)
    return image

preprocessed_image = preprocess_image("sample_image.png")
cv2.imwrite("preprocessed_image.png", preprocessed_image)
</code>
</pre>

### 2. Text Extraction

Extract text from preprocessed images using Tesseract OCR.
<pre>
<strong style="background-color:#2d2d2d; color:#ffffff; padding: 8px; border-radius: 6px;">📄 main.py</strong>
<code>
import pytesseract
from pytesseract import Output

def extract_text(image_path):
    text = pytesseract.image_to_string(image_path, lang='eng')
    return text

text = extract_text("preprocessed_image.png")
print(text)
</code>
</pre>

### 3. Document Categorization

Categorize documents using a fine-tuned Transformer model.
<pre>
<strong style="background-color:#2d2d2d; color:#ffffff; padding: 8px; border-radius: 6px;">📄 main.py</strong>
<code>
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load model and tokenizer
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Tokenize input text
def categorize_document(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    predicted_class = torch.argmax(outputs.logits, dim=1).item()
    return predicted_class

category = categorize_document(text)
print("Document category:", category)

</code>
</pre>

### 4. Summarization

Summarize extracted text for key insights.
<pre>
<strong style="background-color:#2d2d2d; color:#ffffff; padding: 8px; border-radius: 6px;">📄 main.py</strong>
<code>
from transformers import pipeline

# Load summarization pipeline
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def summarize_text(text):
    summary = summarizer(text, max_length=50, min_length=25, do_sample=False)
    return summary[0]['summary_text']

summary = summarize_text(text)
print("Summary:", summary)
</code>
</pre>


### 5. Evaluation and Optimization

Evaluate the model's performance using classification metrics.
<pre>
<strong style="background-color:#2d2d2d; color:#ffffff; padding: 8px; border-radius: 6px;">📄 main.py</strong>
<code>
from sklearn.metrics import classification_report

def evaluate_model(y_true, y_pred):
    report = classification_report(y_true, y_pred, target_names=["Application", "Identity Document", "Financial Document", "Receipt"])
    print(report)

# Example
y_true = [0, 1, 2, 3]  # True labels
y_pred = [0, 1, 2, 3]  # Predicted labels
evaluate_model(y_true, y_pred)
</code>
</pre>
