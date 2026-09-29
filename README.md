# Plant Disease Detection Using CNN
### AI-Powered Plant Leaf Disease Identification System

A complete, modern, and professional web application for detecting plant leaf diseases using Convolutional Neural Networks (CNN) and Flask. Built for college academic project demonstrations, viva presentations, and practical agricultural AI research.

---

## 1. Project Introduction

Plant diseases inflict substantial economic and food security losses annually. Early, accurate detection enables targeted, eco-friendly interventions before widespread crop failure occurs. 

This project provides an automated, computer-vision-based diagnostic platform:
- Users can upload leaf photographs through an intuitive drag-and-drop web portal.
- A deep Convolutional Neural Network (CNN) analyzes the spatial morphology, color patterns, and lesion margins.
- The web app returns the predicted disease class, confidence percentage, clinical description, observable symptoms, and general cultural/organic management advice.

---

## 2. Key Features

- **AI-Powered Diagnostics**: Deep CNN architecture with Conv2D, Batch Normalization, Dropout, and Softmax classification.
- **Fast & Interactive Prediction**: Drag-and-drop upload zone, instant image preview, preloaded demo samples, and sub-second inference.
- **Empirical Performance Dashboard**: Real training and validation accuracy/loss curves powered by Chart.js, plus a validation Confusion Matrix.
- **Dynamic Dataset Adaptation**: The training pipeline automatically scans class subdirectories in `dataset/` without hardcoding class names.
- **Agricultural Pathology Encyclopedia**: Interactive disease knowledge base with instant search filtering.
- **Safe Academic Operation**: The website runs smoothly and shows clear guidance even before the model is trained.
- **Responsive & Modern Design**: Natural green theme, clean typography (Outfit & Plus Jakarta Sans), smooth micro-animations, and mobile hamburger navigation.

---

## 3. Technologies Used

- **Frontend**: HTML5, CSS3, JavaScript (Vanilla ES6+), Chart.js
- **Backend**: Python Flask 3.x, Werkzeug (Secure File Handling)
- **Deep Learning**: TensorFlow / Keras 2.x (Sequential CNN)
- **Image Processing**: Pillow (PIL), NumPy
- **Icons & Typography**: Google Fonts (Outfit, Plus Jakarta Sans), Lucide SVG Icons

---

## 4. Installation Steps

### Step 1: Open Terminal in Project Directory
Navigate to the root directory where `app.py` is located:
```bash
cd "c:\Users\Acer\Desktop\PLANT PROJECT"
```

### Step 2: Install Required Dependencies
Run:
```bash
pip install -r requirements.txt
```

> **Note**: If TensorFlow is not already installed on your system, `pip install -r requirements.txt` will install `tensorflow`, `flask`, `pillow`, `numpy`, and `scikit-learn`.

---

## 5. Dataset Placement

Place your plant leaf image folders inside the `dataset/` directory. Each subfolder represents a class:

```text
dataset/
├── Healthy/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── Tomato_Early_Blight/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── Disease_Class_2/
    ├── image1.jpg
    ├── image2.jpg
    └── ...
```

### Guidelines:
- You can use any popular dataset such as **PlantVillage** or **New Plant Diseases Dataset**.
- Any number of folders can be added. The system automatically detects all folder names.
- Supported file extensions: `.jpg`, `.jpeg`, `.png`, `.webp`.

---

## 6. How to Train the CNN Model

Once your images are placed in `dataset/`, train the model by running:

```bash
python train_model.py
```

### What `train_model.py` does:
1. Discovers all class folders in `dataset/`.
2. Splits data into 80% training and 20% validation.
3. Augments images (random flips, rotations, zoom).
4. Trains the 4-block CNN model with EarlyStopping and Learning Rate reduction.
5. Saves the trained model to: `models/plant_disease_model.keras`.
6. Saves the detected classes to: `models/class_names.json`.
7. Saves performance history and confusion matrix to: `models/training_history.json`.

> **Instant Demo Mode**: If you want to test the full training pipeline before acquiring large datasets, simply run:
> ```bash
> python train_model.py --demo
> ```
> This automatically creates a starter demo dataset with sample leaves and trains the model end-to-end.

---

## 7. How to Start the Flask Server

Start the local development server:

```bash
python app.py
```

Then open your browser and navigate to:
```text
http://127.0.0.1:5000
```

---

## 8. How to Use the Website

1. **Home Page (`/`)**: View project summary, architecture features, and launch detection.
2. **Prediction Page (`/prediction`)**:
   - Drag and drop or browse for a plant leaf photo (or click one of the quick test sample buttons).
   - Click **Detect Disease**.
   - Review the result card showing the predicted disease name, confidence score bar, symptoms, and non-chemical cultural recommendations.
3. **Model Performance Page (`/performance`)**:
   - View genuine training accuracy, validation accuracy, loss metrics, and interactive Chart.js graphs.
   - Inspect the validation Confusion Matrix.
   - If the model is not trained yet, it cleanly displays a training notice instead of showing fake data.
4. **Disease Encyclopedia (`/diseases`)**:
   - Browse diseases and use the live search bar to find symptoms and preventive management.
5. **How It Works (`/how-it-works`)**:
   - Visual breakdown of the CNN layer architecture for academic viva/presentation.
6. **About Page (`/about`)**:
   - Academic motivation, objectives, and viva defense Q&A.

---

## 9. Project Folder Structure

```text
PLANT PROJECT/
│
├── app.py                      # Flask backend application & prediction routes
├── train_model.py              # CNN model training script & evaluation
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
│
├── dataset/                    # Training dataset directory (add class folders here)
│   └── README.md
│
├── models/                     # Saved model artifacts
│   ├── plant_disease_model.keras   (Created after training)
│   ├── class_names.json            (Created after training)
│   ├── training_history.json       (Created after training)
│   └── performance_metrics.json    (Created after training)
│
├── uploads/                    # Temporary user uploads for prediction
│
├── static/
│   ├── css/
│   │   └── style.css           # Agro-AI modern theme styling
│   ├── js/
│   │   └── script.js           # Drag-drop, AJAX inference, Chart.js logic
│   └── images/
│       ├── hero-plant.jpg      # AI Plant visual banner
│       ├── sample_early_blight.jpg
│       └── sample_healthy.jpg
│
└── templates/
    ├── base.html               # Shared layout, header, navbar, footer
    ├── index.html              # Home page
    ├── about.html              # About project & viva preparation
    ├── how_it_works.html       # CNN architecture & workflow steps
    ├── diseases.html           # Disease encyclopedia catalog
    ├── prediction.html         # Main leaf upload & prediction interface
    └── performance.html        # Model accuracy/loss charts & confusion matrix
```

---

## 10. Security & Quality Standards

- Safe filename sanitization using `werkzeug.utils.secure_filename`.
- Strict file type verification (`JPG`, `JPEG`, `PNG`, `WEBP`).
- Max payload size enforcement (16 MB).
- No hardcoded accuracy or fake predictions.
- Non-chemical, informational recommendations strictly compliant with safe agricultural guidance.
