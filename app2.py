import gspread
from oauth2client.service_account import ServiceAccountCredentials
from flask import Flask, render_template
import joblib

import os
from dotenv import load_dotenv

load_dotenv()

GOOGLE_SHEETS_API_KEY = os.getenv("GOOGLE_SHEETS_API_KEY")

# Load the trained spam classifier model and vectorizer
model = joblib.load('spam_classifier_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

app = Flask(__name__)

# Google Sheets Setup
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
creds = ServiceAccountCredentials.from_json_keyfile_name("google_sheets_api.json", scope)
client = gspread.authorize(creds)

# Replace 'your_google_sheet_id' with your actual Google Sheets ID
sheet = client.open_by_key(GOOGLE_SHEETS_API_KEY).sheet1

@app.route('/')
def index():
    # Fetch all data from Google Sheets
    data = sheet.get_all_values()

    # Extract columns
    questions = []
    responses = []
    results = []
    x = 0

    for row in data:
        if len(row) >= 2:
            question = row[0]
            response = row[1]
            
            # Vectorize the AI's response
            vectorized_response = vectorizer.transform([response])
            
            # Predict the class of the AI's response
            prediction = model.predict(vectorized_response)[0]
            classification = "AI PASS" if prediction == 1 else "AI FAIL"
            
            # Store results
            questions.append(question)
            responses.append(response)
            results.append((question, response, classification))
            
            if prediction == 1:
                x += 1
    
    # Calculate MU Score
    mu_score = f"{x}/{len(data)}"

    return render_template('result2.html', results=results, mu_score=mu_score)

if __name__ == '__main__':
    app.run(debug=True)
