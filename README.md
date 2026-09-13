# Personalized Medical Recommendation System (Educational Demo)

An educational machine-learning project that uses an SVM classifier to map a small set of symptom inputs to a broad example condition category.

## Technologies
- Python
- Pandas
- Scikit-learn
- SVM
- Joblib

## How it works
1. Load the example symptom dataset.
2. Split the data into training and test sets.
3. Scale features.
4. Train a linear Support Vector Machine.
5. Save the trained model.
6. Accept symptom inputs from the user.
7. Display the predicted category and general, non-prescriptive advice.

## Run
```bash
pip install -r requirements.txt
python train_model.py
python app.py
```

## Important limitation
This repository is an educational demonstration using a tiny synthetic dataset. It is **not a medical diagnostic or prescription system**. It must not be used to select medicines or make healthcare decisions. Real clinical systems require validated datasets, clinical oversight, safety testing, privacy controls, and regulatory compliance.

## Resume description
**Medical Recommendation System:** Built an educational SVM-based machine-learning application that classifies symptom patterns into broad condition categories and presents non-prescriptive health guidance.
