# Diabetes Risk Prediction App 🩺

A Streamlit web app that predicts diabetes risk from user-entered health
metrics using a pre-trained K-Nearest Neighbors (KNN) model.

## Features

- 🩺 Simple, single-page form for entering 8 health metrics
- 🔍 Risk prediction (High / Low) with a confidence percentage
- 📊 Animated progress bar with staged status messages while "analyzing"
- 📱 Responsive layout — 4 input columns on desktop, reflowing to 2 on
  tablets and 1 on phones
- 🎨 Clean, fixed "Medical Blue" theme with a forced light color scheme so
  text stays readable regardless of the visitor's browser/OS dark-mode
  setting
- ⚠️ Built-in medical disclaimer
- 🚫 No sidebar — kept intentionally minimal, with the "About" note inline
  above the form

## Demo

1. Open the app.
2. Enter your health metrics in the form (Pregnancies, Glucose, Blood
   Pressure, Skin Thickness, Insulin, BMI, Diabetes Pedigree Function, Age).
3. Click **Predict Risk 🔍**.
4. View the predicted risk level and confidence score.

## Requirements

- Python 3.9+
- [Streamlit](https://streamlit.io/)
- [scikit-learn](https://scikit-learn.org/)
- [joblib](https://joblib.readthedocs.io/)
- NumPy

### `requirements.txt`

```
streamlit
scikit-learn
joblib
numpy
```

## Installation

```bash
git clone <your-repo-url>
cd <your-repo-folder>
pip install -r requirements.txt
```

## Model files

This app expects two pre-trained, pickled artifacts in the same directory
as `app.py`:

| File | Purpose |
|---|---|
| `knn_best_diabetes_model.pkl` | Trained KNN classifier (via `joblib`) |
| `scaler.pkl` | `StandardScaler` fitted on the same training data used for the model, so inference-time inputs are scaled consistently |

Both are expected to work with an 8-feature input vector in this exact
order:

```
[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI,
 DiabetesPedigreeFunction, Age]
```

If you don't have these files, you can train your own model (e.g. on the
[Pima Indians Diabetes
Dataset](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database))
and export both with:

```python
import joblib
joblib.dump(model, 'knn_best_diabetes_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
```

If either file is missing or fails to load, the app shows an error and
stops rather than crashing.

## Usage

Run the app locally with:

```bash
streamlit run app.py
```

Then open the URL Streamlit prints (typically `http://localhost:8501`) in
your browser.

## Input fields

| Field | Range | Default |
|---|---|---|
| 🤰 Pregnancies | 0–20 | 0 |
| 🍬 Glucose Level (mg/dL) | 0–300 | 120 |
| 🩸 Blood Pressure (mm Hg) | 0–200 | 70 |
| 📏 Skin Thickness (mm) | 0–100 | 20 |
| 💉 Insulin Level (µU/mL) | 0–900 | 80 |
| 📏 BMI (kg/m²) | 0.0–100.0 | 30.0 |
| 📊 Diabetes Pedigree Function | 0.0–2.5 | 0.5 |
| 🎂 Age (years) | 0–120 | 25 |

## Project structure

```
.
├── app.py                          # Main Streamlit application
├── knn_best_diabetes_model.pkl     # Trained KNN model (not included — add your own)
├── scaler.pkl                      # Fitted StandardScaler (not included — add your own)
├── requirements.txt
└── README.md
```

## Troubleshooting

**"Error loading the model"**
Make sure both `knn_best_diabetes_model.pkl` and `scaler.pkl` exist in the
app's working directory and were saved with a `joblib`/scikit-learn version
compatible with what's installed (pin versions in `requirements.txt` if
you hit unpickling errors).

**Prediction looks off / always the same result**
Check that the feature order used when fitting the scaler and model
matches the order used in `app.py`
(`Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age`).

## Disclaimer

This tool provides an estimate based on the input data and is **not** a
substitute for professional medical advice. Consult a healthcare provider
for proper diagnosis and treatment.

## License

© 2026 Muindi. All rights reserved.
