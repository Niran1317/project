import os
import joblib

FEATURES = ["fever", "cough", "headache", "fatigue", "nausea", "body_pain"]

GENERAL_ADVICE = {
    "Flu": "Rest, drink adequate fluids, and consider medical advice if symptoms are severe or persistent.",
    "Viral_Fever": "Rest and maintain hydration. Seek professional medical advice for persistent or worsening fever.",
    "Common_Cold": "Rest, fluids, and symptom monitoring may help. Consult a clinician if symptoms worsen.",
    "Migraine": "Rest in a quiet environment and seek professional advice for recurrent or severe headaches.",
    "Allergy": "Avoid known triggers and seek professional advice if symptoms are severe."
}

def ask_features():
    values = []
    for feature in FEATURES:
        while True:
            value = input(f"{feature.replace('_', ' ').title()}? (y/n): ").strip().lower()
            if value in ("y", "n"):
                values.append(1 if value == "y" else 0)
                break
            print("Please enter y or n.")
    return [values]

if not os.path.exists("medical_svm.joblib"):
    print("Model not found. Run: python train_model.py")
    raise SystemExit(1)

model = joblib.load("medical_svm.joblib")
print("Educational symptom classifier")
print("--------------------------------")
print("This is NOT a medical diagnosis or prescription tool.")

sample = ask_features()
prediction = model.predict(sample)[0]
probabilities = model.predict_proba(sample)[0]
confidence = max(probabilities)

print("\nModel output:", prediction)
print("Model confidence on this demo classifier:", round(confidence * 100, 2), "%")
print("General information:", GENERAL_ADVICE.get(prediction, "Consult a qualified healthcare professional."))

print("\nImportant: This demo uses a tiny synthetic dataset and must not be used for real medical decisions.")
