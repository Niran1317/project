# Age & Gender Detection using OpenCV

A computer-vision project that detects faces from a webcam and predicts an approximate age group and gender using pretrained OpenCV DNN models.

## Technologies
- Python
- OpenCV
- NumPy
- Deep Neural Network (DNN)

## Project structure
```text
Age-Gender-Detection/
├── main.py
├── requirements.txt
├── models/
│   └── README.md
├── sample/
└── .gitignore
```

## Setup
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

pip install -r requirements.txt
```

Place the six required pretrained model files inside `models/`. See `models/README.md`.

## Run
```bash
python main.py
```
Press `Q` to exit.

## Important note
Age and gender predictions are approximate model outputs and should not be treated as verified personal attributes.

## Resume description
**Age & Gender Detection:** Developed a computer-vision application using OpenCV DNN models to detect faces and estimate age groups and gender from a live webcam feed.
