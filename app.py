from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load the full pipeline and model from the saved file
with open('full_pipeline.pkl', 'rb') as file:
    loaded_pipeline, loaded_model = pickle.load(file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        user_input = request.form['user_input']

        # Preprocess user input
        user_input_preprocessed = loaded_pipeline.transform([user_input])

        # Make prediction
        prediction = loaded_model.predict(user_input_preprocessed)[0]

        return render_template('index.html', prediction=prediction, user_input=user_input)

if __name__ == '__main__':
    app.run(debug=True)
