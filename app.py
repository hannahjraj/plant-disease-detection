"""
Plant Disease Detection Web Application
Flask Backend Server
AI-Powered Plant Leaf Disease Identification System Using CNN
"""

import os
import re
import json
import logging
from datetime import datetime
from werkzeug.utils import secure_filename
from flask import Flask, render_template, request, jsonify, send_from_directory, url_for

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Base Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
MODEL_PATH = os.path.join(MODELS_DIR, "plant_disease_model.keras")
CLASS_NAMES_PATH = os.path.join(MODELS_DIR, "class_names.json")
HISTORY_PATH = os.path.join(MODELS_DIR, "training_history.json")
METRICS_PATH = os.path.join(MODELS_DIR, "performance_metrics.json")

# Ensure required directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB max limit
app.config["ALLOWED_EXTENSIONS"] = {"png", "jpg", "jpeg", "webp"}

# Global model cache
cached_model = None
cached_classes = None

# Comprehensive Agricultural Knowledge Base for Plant Classes
DISEASE_KNOWLEDGE_BASE = {
    "healthy": {
        "title": "Healthy Leaf",
        "plant": "General Plant",
        "status": "Healthy",
        "badge_color": "green",
        "description": "The leaf exhibits balanced chlorophyll levels, uniform surface texture, and no visible fungal or bacterial lesions.",
        "symptoms": [
            "Vibrant, uniform green coloration",
            "Smooth leaf surface without lesions or dark spots",
            "Firm, intact leaf margin and turgid stem",
            "Normal photosynthetic vein pattern"
        ],
        "recommendations": [
            "Maintain current regular watering schedule and avoid waterlogging.",
            "Ensure proper sunlight exposure suitable for the crop variety.",
            "Continue routine inspections of lower leaves for early pest detection.",
            "Apply balanced organic compost to maintain soil microbial health."
        ]
    },
    "early_blight": {
        "title": "Early Blight (Alternaria solani)",
        "plant": "Tomato / Potato",
        "status": "Disease Detected",
        "badge_color": "amber",
        "description": "A common fungal disease caused by Alternaria solani, characterized by dark concentric rings forming a bullseye pattern on older foliage.",
        "symptoms": [
            "Brown-to-black spots with concentric target-like rings",
            "Yellow halo surrounding the circular necrotic lesions",
            "Lower and older leaves developing symptoms first",
            "Premature defoliation exposing fruit to sunscald"
        ],
        "recommendations": [
            "Prune and safely dispose of affected bottom leaves to improve airflow.",
            "Avoid overhead irrigation; water directly at the base or root zone.",
            "Apply approved organic copper-based or bio-fungicide sprays at early onset.",
            "Practice crop rotation by avoiding planting Solanaceae crops consecutively."
        ]
    },
    "late_blight": {
        "title": "Late Blight (Phytophthora infestans)",
        "plant": "Tomato / Potato",
        "status": "Disease Detected",
        "badge_color": "red",
        "description": "A destructive water-mold (oomycete) pathogen that thrives in cool, moist weather and can devastate whole canopies rapidly.",
        "symptoms": [
            "Water-soaked dark, irregularly shaped lesions on leaves and stems",
            "White velvety fungal growth on the undersides of wet leaves",
            "Rapid browning and collapse of green tissues in humid conditions",
            "Brown firm rot spreading into developing fruits or tubers"
        ],
        "recommendations": [
            "Remove and destroy severely infected plant debris immediately (do not compost).",
            "Space plants adequately to facilitate rapid drying of wet leaves.",
            "Apply preventive bio-protective sprays before extended wet or foggy weather.",
            "Choose certified disease-resistant seed and plant varieties."
        ]
    },
    "leaf_mold": {
        "title": "Leaf Mold (Passalora fulva)",
        "plant": "Tomato",
        "status": "Disease Detected",
        "badge_color": "amber",
        "description": "A fungal foliar disease prevalent in high humidity environments and greenhouses, causing pale spots and moldy undersides.",
        "symptoms": [
            "Pale greenish-yellow chlorotic spots on the upper leaf surface",
            "Olive-green to velvety brown mold growth directly underneath",
            "Infected leaves curl, wither, and drop prematurely",
            "Reduced fruit yield due to decreased photosynthetic area"
        ],
        "recommendations": [
            "Increase greenhouse ventilation and lower relative humidity below 80%.",
            "Provide ample spacing between plants for adequate cross-breeze.",
            "Water early in the morning so plant foliage dries quickly.",
            "Sanitize trellises, stakes, and greenhouse walls between planting seasons."
        ]
    },
    "septoria_leaf_spot": {
        "title": "Septoria Leaf Spot (Septoria lycopersici)",
        "plant": "Tomato",
        "status": "Disease Detected",
        "badge_color": "amber",
        "description": "Fungal infection causing abundant circular spots with dark brown margins and gray centers containing tiny black fruiting bodies.",
        "symptoms": [
            "Numerous small circular spots (2-3mm) with dark borders",
            "Ash-grey center with tiny black specks (pycnidia)",
            "Yellowing around spots leading to leaf shedding from bottom upwards",
            "Stem lesions when infection becomes severe"
        ],
        "recommendations": [
            "Mulch heavily around the plant base to prevent soil splash onto foliage.",
            "Remove lower infected leaves as soon as first spots appear.",
            "Clean gardening shears with 70% alcohol between trimming different plants.",
            "Rotate crops annually with non-solanaceous crops."
        ]
    },
    "bacterial_spot": {
        "title": "Bacterial Spot (Xanthomonas)",
        "plant": "Pepper / Tomato",
        "status": "Disease Detected",
        "badge_color": "red",
        "description": "Bacterial plant infection causing water-soaked spots that turn dark brown or black with a translucent or yellow halo.",
        "symptoms": [
            "Small, dark angular spots that appear greasy or water-soaked",
            "Lesions turning scabby or blister-like on leaf surfaces",
            "Yellowing and severe defoliation during warm, rainy weather",
            "Rough, raised brown specks on fruit skins"
        ],
        "recommendations": [
            "Use certified pathogen-free seeds and disease-resistant cultivars.",
            "Avoid handling plants or cultivating fields while foliage is wet.",
            "Apply copper-based sprays preventively during warm, rainy periods.",
            "Implement a minimum 2-year crop rotation schedule."
        ]
    },
    "yellow_leaf_curl": {
        "title": "Yellow Leaf Curl Virus (TYLCV)",
        "plant": "Tomato",
        "status": "Disease Detected",
        "badge_color": "red",
        "description": "A viral disease transmitted by the silverleaf whitefly, causing stunted growth, cupping, and yellowing of young leaves.",
        "symptoms": [
            "Severe upward curling and cupping of leaflets",
            "Interveinal yellowing (chlorosis) and reduced leaf size",
            "Marked plant stunting with an erect, bushy appearance",
            "Significant flower abortion and drastically reduced fruit set"
        ],
        "recommendations": [
            "Control whitefly insect vectors using yellow sticky traps and neem oil.",
            "Use 50-mesh insect netting in greenhouses and nurseries.",
            "Promptly rogue out and destroy severely symptomatic young plants.",
            "Plant virus-resistant hybrid varieties suited for your region."
        ]
    },
    "apple_scab": {
        "title": "Apple Scab (Venturia inaequalis)",
        "plant": "Apple",
        "status": "Disease Detected",
        "badge_color": "amber",
        "description": "One of the most widespread apple fungal diseases, attacking leaves and fruit surfaces during humid spring weather.",
        "symptoms": [
            "Dull olive-green velvety spots turning dark brown or black",
            "Leaves becoming distorted, puckered, and dropping early",
            "Cork-like scabby spots on fruit skins that may crack",
            "Twig lesions in severe recurring infections"
        ],
        "recommendations": [
            "Rake and compost fallen leaves in autumn to eliminate overwintering spores.",
            "Prune tree canopy to maximize sunlight penetration and air circulation.",
            "Apply sulfur or bio-fungicide sprays from early bud break to petal fall.",
            "Select scab-resistant apple cultivars when establishing new orchards."
        ]
    },
    "rust": {
        "title": "Plant Rust (Puccinia spp.)",
        "plant": "Corn / Beans / Cereals",
        "status": "Disease Detected",
        "badge_color": "amber",
        "description": "Fungal pathogen recognized by powdery rust-colored pustules on leaf undersides that rupture and release infectious spores.",
        "symptoms": [
            "Cinnamon-brown to golden-orange powdery pustules on foliage",
            "Chlorotic flecks that rapidly expand into spore-producing blisters",
            "Leaves yellowing and drying out as pustules coalesce",
            "Premature crop maturity and reduced grain/seed weight"
        ],
        "recommendations": [
            "Select rust-tolerant or resistant hybrid varieties.",
            "Maintain optimal plant spacing to reduce micro-climate humidity.",
            "Remove alternate weed hosts bordering the field.",
            "Apply preventive bio-control agents at initial symptom emergence."
        ]
    }
}

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in app.config["ALLOWED_EXTENSIONS"]

def format_class_name(raw_name):
    """Converts raw folder names like 'Tomato___Early_blight' into clean human titles."""
    clean = raw_name.replace("___", " - ").replace("__", " - ").replace("_", " ")
    # Capitalize words
    return " ".join([word.capitalize() for word in clean.split()])

def resolve_disease_metadata(raw_class_name):
    """Matches class name to the knowledge base or generates a clean fallback."""
    raw_lower = raw_class_name.lower().replace("_", " ").replace("-", " ")
    
    # 1. Check for healthy
    if "healthy" in raw_lower:
        base = DISEASE_KNOWLEDGE_BASE["healthy"].copy()
        base["title"] = format_class_name(raw_class_name)
        return base

    # 2. Check for known disease patterns
    for key, data in DISEASE_KNOWLEDGE_BASE.items():
        if key in raw_lower:
            matched = data.copy()
            matched["title"] = format_class_name(raw_class_name)
            return matched

    # 3. Fallback for custom or novel class detected
    return {
        "title": format_class_name(raw_class_name),
        "plant": "Foliage Crop",
        "status": "Disease Detected" if "healthy" not in raw_lower else "Healthy",
        "badge_color": "amber" if "healthy" not in raw_lower else "green",
        "description": f"Classification result for {format_class_name(raw_class_name)} based on trained CNN feature maps.",
        "symptoms": [
            "Foliar irregularities detected by deep feature extraction",
            "Discoloration or textural variance on leaf lamina",
            "Possible lesion margins identified across receptive fields"
        ],
        "recommendations": [
            "Isolate suspected plants to monitor disease progression.",
            "Consult local agricultural extension service for specialized confirmation.",
            "Maintain proper sanitation and avoid excessive foliage moisture."
        ]
    }

def get_model_status():
    """Returns the current state of model training and metadata."""
    is_trained = os.path.exists(MODEL_PATH)
    classes = []
    history = {}
    metrics = {}

    if os.path.exists(CLASS_NAMES_PATH):
        try:
            with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
                classes = json.load(f)
        except Exception as e:
            logger.warning(f"Could not read class names: {e}")

    if os.path.exists(HISTORY_PATH):
        try:
            with open(HISTORY_PATH, "r", encoding="utf-8") as f:
                history = json.load(f)
        except Exception as e:
            logger.warning(f"Could not read history: {e}")

    if os.path.exists(METRICS_PATH):
        try:
            with open(METRICS_PATH, "r", encoding="utf-8") as f:
                metrics = json.load(f)
        except Exception as e:
            logger.warning(f"Could not read metrics: {e}")

    return {
        "trained": is_trained,
        "model_path": MODEL_PATH if is_trained else None,
        "classes": classes,
        "history": history,
        "metrics": metrics
    }

def load_prediction_model():
    """Safely loads or retrieves the cached Keras model."""
    global cached_model, cached_classes
    if not os.path.exists(MODEL_PATH):
        return None, None

    try:
        if cached_model is None:
            logger.info("Loading Keras model into memory...")
            import tensorflow as tf
            cached_model = tf.keras.models.load_model(MODEL_PATH)
            logger.info("Model loaded successfully.")

        if cached_classes is None and os.path.exists(CLASS_NAMES_PATH):
            with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
                cached_classes = json.load(f)

        return cached_model, cached_classes
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return None, None

# -------------------------------------------------------------
# Web Routes
# -------------------------------------------------------------

@app.route("/")
def index():
    status = get_model_status()
    return render_template("index.html", status=status)

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/how-it-works")
def how_it_works():
    return render_template("how_it_works.html")

@app.route("/prediction")
def prediction():
    status = get_model_status()
    return render_template("prediction.html", status=status)

@app.route("/diseases")
def diseases():
    status = get_model_status()
    
    # Compile dynamic disease cards
    all_cards = []
    if status["classes"]:
        for raw in status["classes"]:
            meta = resolve_disease_metadata(raw)
            meta["raw_class"] = raw
            all_cards.append(meta)
    else:
        # Provide representative educational catalog
        for key, data in DISEASE_KNOWLEDGE_BASE.items():
            card = data.copy()
            card["raw_class"] = key
            all_cards.append(card)

    return render_template("diseases.html", status=status, disease_cards=all_cards)

@app.route("/performance")
def performance():
    status = get_model_status()
    has_training_data = bool(status["trained"] and status["history"])
    return render_template(
        "performance.html",
        status=status,
        has_training_data=has_training_data,
        history=status.get("history", {}),
        metrics=status.get("metrics", {})
    )

@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)

# -------------------------------------------------------------
# API Endpoints
# -------------------------------------------------------------

@app.route("/api/model-status", methods=["GET"])
def api_model_status():
    return jsonify(get_model_status())

@app.route("/predict", methods=["POST"])
def predict():
    """Handles plant leaf image prediction."""
    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        return jsonify({
            "success": False,
            "model_ready": False,
            "message": "Model not trained yet. Please train the CNN model using train_model.py before making predictions."
        }), 200

    # Validate image file in request
    if "file" not in request.files:
        # Check if sample image request was passed
        sample_name = request.form.get("sample_image")
        if sample_name:
            sample_path = os.path.join(BASE_DIR, "static", "images", secure_filename(sample_name))
            if os.path.exists(sample_path):
                file_path = sample_path
                web_url = url_for("static", filename=f"images/{sample_name}")
            else:
                return jsonify({"success": False, "message": "Selected sample image not found."}), 400
        else:
            return jsonify({"success": False, "message": "No image file provided in request."}), 400
    else:
        file = request.files["file"]
        if file.filename == "":
            return jsonify({"success": False, "message": "No file was selected for upload."}), 400

        if not allowed_file(file.filename):
            return jsonify({
                "success": False,
                "message": "Unsupported file format. Please upload a JPG, JPEG, PNG, or WEBP image."
            }), 400

        try:
            filename = secure_filename(file.filename)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            saved_filename = f"leaf_{timestamp}_{filename}"
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], saved_filename)
            file.save(file_path)
            web_url = url_for("uploaded_file", filename=saved_filename)
        except Exception as e:
            logger.error(f"Upload error: {e}")
            return jsonify({"success": False, "message": "Failed to save uploaded image."}), 500

    # Image Preprocessing & Model Inference
    try:
        from PIL import Image
        import numpy as np

        # Open and validate image
        try:
            img = Image.open(file_path)
            img.verify()
            img = Image.open(file_path)  # Reopen after verify
        except Exception:
            return jsonify({
                "success": False,
                "message": "The uploaded file is corrupted or not a valid image."
            }), 400

        # Load CNN Model
        model, class_names = load_prediction_model()
        if model is None or not class_names:
            return jsonify({
                "success": False,
                "model_ready": False,
                "message": "CNN model or class labels could not be loaded. Please ensure train_model.py finished successfully."
            }), 200

        # Image Preprocessing matching CNN input (128x128 RGB)
        img_rgb = img.convert("RGB")
        target_size = (128, 128)
        img_resized = img_rgb.resize(target_size)
        img_array = np.array(img_resized, dtype=np.float32)
        # Note: If Rescaling layer is part of CNN, keep 0-255 or 0-1 depending on architecture
        # Our model in train_model.py has a Rescaling(1./255) layer built-in
        input_tensor = np.expand_dims(img_array, axis=0)

        # Run Prediction
        predictions = model.predict(input_tensor, verbose=0)
        predicted_idx = int(np.argmax(predictions[0]))
        confidence = float(predictions[0][predicted_idx]) * 100.0

        if predicted_idx < len(class_names):
            predicted_class = class_names[predicted_idx]
        else:
            predicted_class = "Unknown"

        meta = resolve_disease_metadata(predicted_class)

        # Build top-3 predictions for extra analytical depth
        top_indices = np.argsort(predictions[0])[::-1][:min(3, len(class_names))]
        top_predictions = []
        for idx in top_indices:
            top_predictions.append({
                "class_name": format_class_name(class_names[idx]),
                "confidence": round(float(predictions[0][idx]) * 100.0, 2)
            })

        return jsonify({
            "success": True,
            "model_ready": True,
            "raw_class": predicted_class,
            "disease_name": meta["title"],
            "plant": meta["plant"],
            "status": meta["status"],
            "badge_color": meta["badge_color"],
            "confidence": round(confidence, 2),
            "description": meta["description"],
            "symptoms": meta["symptoms"],
            "recommendations": meta["recommendations"],
            "image_url": web_url,
            "top_predictions": top_predictions
        })

    except Exception as e:
        logger.error(f"Inference error: {e}")
        return jsonify({
            "success": False,
            "message": "An error occurred while processing the image with the CNN model."
        }), 500

@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({
        "success": False,
        "message": "File size exceeds the 16MB limit. Please upload a smaller image."
    }), 413

@app.errorhandler(404)
def page_not_found(e):
    return render_template("base.html", not_found=True), 404

if __name__ == "__main__":
    print("=" * 65)
    print("     PLANT DISEASE DETECTION WEB APPLICATION (FLASK)")
    print("  Server starting on: http://127.0.0.1:5000")
    print("=" * 65)
    app.run(debug=True, host="127.0.0.1", port=5000)
