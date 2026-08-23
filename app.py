from flask import Flask, render_template, request
import numpy as np
import pickle
import os

app = Flask(__name__)

# Load models & accuracies
models = pickle.load(open('model.pkl', 'rb'))
accuracy = pickle.load(open('accuracy.pkl', 'rb'))

# Find the best model
best_model = max(accuracy, key=accuracy.get)

# Home route
@app.route('/')
def home():
    return render_template('index.html',
                           best_model=best_model,
                           accuracy=accuracy)

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = request.form['sonar_data']

        # Split input by commas
        input_list = input_data.split(',')

        # Validate length
        if len(input_list) != 60:
            return render_template('index.html',
                                   prediction_text="Please enter exactly 60 values!",
                                   best_model=best_model,
                                   accuracy=accuracy)

        # Convert to float
        input_float = [float(i) for i in input_list]

        # Convert to NumPy array
        input_array = np.asarray(input_float).reshape(1, -1)

        # Make prediction
        model = models[best_model] # selects best model amon the models
        prediction = model.predict(input_array)

        result = "The object is Rock" if prediction[0] == 'R' else "The object is Mine"

        return render_template('index.html',
                               prediction_text=f"Prediction: {result}",
                               best_model=best_model,
                               accuracy=accuracy)

    except ValueError:
        return render_template('index.html',
                               prediction_text="Invalid input! Please enter numbers only.",
                               best_model=best_model,
                               accuracy=accuracy)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=False)
