import os
import numpy as np
from flask import Flask, render_template, request, redirect, flash
from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename
from PIL import Image, ImageChops, ImageEnhance

app = Flask(__name__)
app.secret_key = "supersecretkey"

model = load_model("best_model_finetuned.h5")

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def convert_to_ela(image_path, quality=90):
    try:
        original   = Image.open(image_path).convert("RGB")
        temp_path  = "temp_ela.jpg"
        original.save(temp_path, "JPEG", quality=quality)
        compressed = Image.open(temp_path).convert("RGB")

        ela_image = ImageChops.difference(original, compressed)
        extrema   = ela_image.getextrema()
        max_diff  = max([ex[1] for ex in extrema])
        if max_diff == 0:
            max_diff = 1
        scale     = 255.0 / max_diff
        ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)
        return ela_image
    except:
        return Image.open(image_path).convert("RGB")

def preprocess_image(filepath):
    orig_img   = Image.open(filepath).convert("RGB").resize((224, 224))
    orig_array = np.array(orig_img) / 255.0
    orig_array = np.expand_dims(orig_array, axis=0)

    ela_img   = convert_to_ela(filepath)
    ela_img   = ela_img.resize((224, 224))
    ela_array = np.array(ela_img) / 255.0
    ela_array = np.expand_dims(ela_array, axis=0)

    return orig_array, ela_array

def predict_image(filepath):
    orig_array, ela_array = preprocess_image(filepath)
    prediction = model.predict([orig_array, ela_array])[0][0]
    if prediction > 0.5:
        result     = "forged"
        confidence = round(float(prediction) * 100, 2)
    else:
        result     = "authentic"
        confidence = round((1 - float(prediction)) * 100, 2)
    return result, confidence

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        files = request.files.getlist("file")

        if not files or files[0].filename == "":
            flash("No file selected.")
            return redirect(request.url)

        results = []
        errors  = []

        for file in files:
            if not allowed_file(file.filename):
                errors.append(f"{file.filename}: Invalid file type.")
                continue
            try:
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
                file.save(filepath)
                result, confidence = predict_image(filepath)
                results.append({
                    "filename"  : filename,
                    "result"    : result,
                    "confidence": confidence
                })
            except Exception as e:
                errors.append(f"{file.filename}: Error — {str(e)}")

        if errors:
            for error in errors:
                flash(error)

        if results:
            
            return render_template(
                "result.html",
                results=results,
                is_batch=len(results) > 1
            )

        return redirect(request.url)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)