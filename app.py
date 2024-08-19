from flask import Flask, render_template, request
import joblib

# Load the trained spam classifier model and vectorizer
model = joblib.load('spam_classifier_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify():
    message = request.form['message']
    data = vectorizer.transform([message])  # Preprocessing the input
    prediction = model.predict(data)
    
    # Make sure this is properly indented and the colon is present
    if prediction == 1:  
        result = "Conversational-AI PASS"
    else:
        result = "Conversational-AI FAIL"
    
    return render_template('result.html', message=message, result=result)

if __name__ == '__main__':
    app.run(debug=True)
