from flask import Flask, render_template, request
import pickle
import os

# CREATE APP
app = Flask(__name__)

# LOAD MODEL & VECTORIZER (safe path)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = pickle.load(open(os.path.join(BASE_DIR, 'model.pkl'), 'rb'))
vectorizer = pickle.load(open(os.path.join(BASE_DIR, 'vectorizer.pkl'), 'rb'))

# HOME PAGE
@app.route('/')
def home():
    return render_template('index.html')

# PREDICTION
@app.route('/predict', methods=['POST'])
def predict():
    news = request.form.get('news', '')

    if news.strip() == "":
        return render_template('index.html', prediction_text="Please enter news text")

    data = vectorizer.transform([news])
    prediction = model.predict(data)[0]

    print("Raw Prediction:", prediction)

    # Clean prediction
    prediction = str(prediction).lower()

    if prediction in ["0", "real"]:
        result = "Real News"
    elif prediction in ["1", "fake"]:
        result = "Fake News"
    else:
        result = "Unknown Result"

    return render_template('index.html', prediction_text=result)

# RUN APP
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)