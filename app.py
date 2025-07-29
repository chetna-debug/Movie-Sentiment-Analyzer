from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load trained model and vectorizer
model = pickle.load(open("sentiment_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    review = request.form['review']
    review_transformed = vectorizer.transform([review])
    prediction = model.predict(review_transformed)[0]
    sentiment = "Positive 😊" if prediction == 1 else "Negative 😞"
    return render_template('index.html', review=review, prediction=sentiment)

if __name__ == '__main__':
    app.run(debug=True)
  
