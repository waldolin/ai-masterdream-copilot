from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load models
model_h5 = load_model('my_model.h5')
model_saved_model = load_model('my_saved_model')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    # Assume we receive a list of numbers as input features
    features = data['features']
    # Use my_model.h5 model for prediction
    prediction_h5 = model_h5.predict([features]).tolist()
    # Use my_saved_model model for prediction
    prediction_saved_model = model_saved_model.predict([features]).tolist()

    # Return prediction results
    return jsonify({
        'prediction_h5': prediction_h5,
        'prediction_saved_model': prediction_saved_model
    })

if __name__ == '__main__':
    app.run(debug=True)