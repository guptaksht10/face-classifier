## 🧠 Face Classifier
This is a Python-based face classification system using OpenCV for face detection and NumPy-based datasets for recognition.

## 📁 Project Structure
```bash
.
├── data/
│   ├── Akshat.npy           # Encoded face data for "Akshat"
│   ├── Mummy.npy            # Encoded face data for "Mummy"
│   └── .ipynb_checkpoints/  # (Ignore - Jupyter artifacts)
│
├── frontalfaceDet.py        # Face detection script using Haar Cascades
├── frontalfaceRecog.py      # Face recognition/classification script
├── haarcascade_frontalface_default.xml  # Haar Cascade model for face detection
├── requirements.txt         # Python dependencies
├── .gitignore               # Git ignore settings
```

## 🚀 How to Run
### 1. Clone the repository

```bash
git clone https://github.com/guptaksht10/face-classifier.git
cd face-classifier
```

### 2. Create and activate virtual environment
```bash
python -m venv venv
source venv/bin/activate        # On macOS/Linux
# OR
venv\Scripts\activate           # On Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run Face Detection
```bash
python frontalfaceDet.py
```

### 5. Run Face Recognition
```bash
python frontalfaceRecog.py
```

## 🧩 Dependencies
These are listed in requirements.txt, but key ones include:

- opencv-python

- numpy

- scikit-learn

## 📦 Notes
- The data/ folder stores face encodings (*.npy files).

- You can add new faces by extracting features and saving them with NumPy.

- Haar Cascade XML file is required for detection (already included).

