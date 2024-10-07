import os
from flask import Flask, request, render_template, redirect, url_for, send_file
import tensorflow as tf
from PIL import Image
import numpy as np
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from reportlab.pdfgen import canvas
import io
from datetime import datetime

app = Flask(__name__)

# Load your pre-trained model
MODEL_PATH = "C:/Users/Ahmad/Downloads/VGG16.h5"  # Replace with your model's path
model = tf.keras.models.load_model(MODEL_PATH)


# Function to preprocess the uploaded image
def preprocess_image(image):
    # Convert to RGB by repeating the grayscale channel into 3 channels
    image = image.convert("RGB")

    # Resize to model input size
    image = image.resize((224, 224))

    # Convert image to array and normalize
    image = np.array(image) / 255.0

    # Add batch dimension
    image = np.expand_dims(image, axis=0)

    return image


@app.route('/', methods=['GET', 'POST'])
@app.route('/', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        # Get form data
        patient_name = request.form['patient_name']
        age = request.form['age']
        sex = request.form['sex']
        doctor_name = request.form['doctor_name']  # Get the doctor's name
        notes = request.form['notes']
        month_year = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        report_name = patient_name + " " + month_year +'.pdf'
        # Get file and process image
        file = request.files['file']
        if file:
            image = Image.open(file)
            processed_image = preprocess_image(image)
            prediction = model.predict(processed_image)
            diagnosis = "Pneumonia" if prediction[0] > 0.5 else "Normal"

            # Generate report with all details, including the doctor's name
            pdf = generate_report(patient_name, age, sex, notes, doctor_name, diagnosis, prediction[0], image)

            return send_file(pdf, as_attachment=True, download_name=report_name)
    return render_template('upload.html')


def generate_report(patient_name, age, sex, notes, doctor_name, diagnosis, confidence, image):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)

    # Set up title and subtitle
    c.setFont("Helvetica-Bold", 20)
    c.drawString(100, 750, "Pneumonia Detection Report")

    c.setFont("Helvetica", 12)
    c.drawString(100, 730, f"Report Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Draw a line separator
    c.setStrokeColor(colors.grey)
    c.line(100, 720, 500, 720)

    # Patient Information Section
    c.setFont("Helvetica-Bold", 14)
    c.drawString(100, 700, "Patient Information")

    c.setFont("Helvetica", 12)
    c.drawString(100, 680, f"Name: {patient_name}")
    c.drawString(100, 660, f"Age: {age}")
    c.drawString(100, 640, f"Sex: {sex}")

    c.line(100,620, 500, 620)

    # Patient Information Section
    c.setFont("Helvetica-Bold", 14)
    c.drawString(100, 600, "Doctor Notes")

    # Doctor's Name
    c.setFont("Helvetica", 12)
    c.drawString(100, 580, f"Doctor: {doctor_name}")

    # Notes Section
    if notes:
        c.setFont("Helvetica", 12)
        c.drawString(100, 560, "Additional Notes:")
        c.setFont("Helvetica-Oblique", 10)
        c.drawString(100, 540, f"{notes}")

    # Draw a line separator
    c.setStrokeColor(colors.grey)
    c.line(100, 520, 500, 520)

    # Diagnosis Information Section
    c.setFont("Helvetica-Bold", 14)
    c.drawString(100, 500, "Diagnosis Information")

    c.setFont("Helvetica", 12)
    c.drawString(100, 480, f"Diagnosis: {diagnosis}")
    #c.drawString(100, 500, f"Confidence: {confidence:.2f}")

    # Draw the X-ray image
    image_path = "temp_xray.png"  # Save image temporarily
    image.save(image_path)
    c.drawImage(image_path, 100, 150, width=350, height=300)

    # Footer Section
    c.setFont("Helvetica", 10)
    c.drawString(100, 100, "This report is auto-generated and should be reviewed by a medical professional.")

    # Final Line Separator
    c.setStrokeColor(colors.grey)
    c.line(100, 90, 460, 90)

    c.showPage()
    c.save()

    buffer.seek(0)
    os.remove(image_path)  # Clean up temporary image file
    return buffer
if __name__ == '__main__':
    app.run(debug=True)
