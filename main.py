from flask import Flask, render_template, request
import os
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

from ocr import extract_text

import pytesseract
from PIL import Image

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Load Dataset
data = pd.read_csv("scam_dataset.csv")
data = data.dropna()
print(data.head())
print(data.columns)
print(data.shape)



# Train Model
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(data["text"])
y = data["label"]

model = MultinomialNB()
model.fit(X, y)


@app.route("/", methods=["GET", "POST"])
def home():

    result = ""
    confidence = 0
    scam_type = "Unknown"

    if request.method == "POST":

        message = request.form.get("message", "")

        image = request.files.get("image")

        if image and image.filename:
            image_path = os.path.join(
                app.config["UPLOAD_FOLDER"],
                image.filename
            )

            image.save(image_path)

            message = extract_text(image_path)

        test = vectorizer.transform([message])
        prediction = model.predict(test)[0]

        print("Message:", message)
        print("Prediction:", prediction)

        probability = model.predict_proba(test).max() * 100

        result = prediction.upper()
        confidence = round(probability, 2)

        text = message.lower()

        if "otp" in text:
            scam_type = "OTP Scam"

        elif "bank" in text:
            scam_type = "Bank Scam"

        elif "lottery" in text or "prize" in text:
            scam_type = "Lottery Scam"

        elif "upi" in text or "payment" in text:
            scam_type = "Payment Scam"

        elif "job" in text or "interview" in text:
            scam_type = "Job Scam"

        elif "gift" in text:
            scam_type = "Gift Scam"

        elif "click" in text:
            scam_type = "Phishing Scam"

    return render_template(
        "index.html",
        result=result,
        confidence=confidence,
        scam_type=scam_type
    )


if __name__ == "__main__":
    app.run(debug=True)