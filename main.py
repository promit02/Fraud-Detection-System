from flask import Flask, request, jsonify
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch
import numpy as np

# Initialize Flask application
app = Flask(__name__)

# Load the pre-trained DistilBERT model and tokenizer
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

# Set the model to evaluation mode
model.eval()

# Define a route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from the request
        data = request.json

        # Preprocess the input data
        features = preprocess_data(data)

        # Tokenize and convert to PyTorch tensors
        inputs = tokenizer(features, return_tensors="pt", truncation=True, padding=True)

        # Make a prediction using the loaded DistilBERT model
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits

        # Convert the logits to probabilities and get the predicted class
        probabilities = torch.nn.functional.softmax(logits, dim=1).numpy()
        predicted_class = np.argmax(probabilities)

        # Convert the prediction to a human-readable format
        result = postprocess_prediction(predicted_class)

        # Return the result as JSON
        return jsonify({'result': result})

    except Exception as e:
        return jsonify({'error': str(e)})

# Define the preprocessing function
def preprocess_data(data):
    # Assuming 'data' is a dictionary containing the input text
    text_input = data.get('text_input', '')

    # Tokenize the input text
    tokenized_input = tokenizer(text_input, return_tensors="pt", truncation=True, padding=True)

    return tokenized_input

# Define the postprocessing function
def postprocess_prediction(predicted_class):
    # Assuming binary classification (0 or 1)
    if predicted_class == 0:
        result = 'Non-Fraudulent'
    else:
        result = 'Fraudulent'

    return result

# Run the Flask application
if __name__ == '__main__':
    app.run(debug=True)

