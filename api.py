from flask import Flask, request, jsonify
import xgboost as xgb
import pandas as pd
from flask_cors import CORS

# Load your XGBoost model
model = xgb.XGBClassifier()
model.load_model('./xgb_class.json')

app = Flask(__name__)
CORS(app)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract data from POST request
        data = request.get_json()
        df = pd.DataFrame(data, index=[0])

        # Make prediction
        prediction = model.predict(df)

        # Return prediction
        return jsonify({'prediction': int(prediction[0])})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=8080)