# app.py
from flask import Flask, request, jsonify
from transformers import pipeline

# 1. Initialize the Flask web application
app = Flask(__name__)

# 2. Load the pre-trained transformer model for translation
# This pipeline handles preprocessing (tokenization) and translation.
# It's a stand-in for the custom-trained model mentioned in the project.
print("Loading translation model...")
translator = pipeline("translation_en_to_es", model="t5-small")
print("Model loaded successfully!")

# 3. Define the translation endpoint for the web interface
@app.route('/translate', methods=['POST'])
def translate_text():
    """
    Receives English text, translates it to Spanish, and returns the result.
    """
    # Get the JSON data from the request
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({"error": "No text provided"}), 400

    english_text = data['text']

    # Use the loaded transformer model to translate the text
    # The model handles complex grammar and context.
    translated_output = translator(english_text, max_length=100)
    
    # Extract the translated text from the model's output
    spanish_text = translated_output[0]['translation_text']
    
    # Return the result as JSON
    return jsonify({
        "original_text": english_text,
        "translated_text": spanish_text
    })

# Main entry point to run the web server
if __name__ == '__main__':
    # Runs the app on localhost, port 5000
    app.run(debug=True, port=5000)
