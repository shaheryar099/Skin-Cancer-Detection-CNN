# 🔬 DermAI — Skin Cancer Detection System

A Final Year Project built with **Streamlit** and **TensorFlow** that uses a CNN (VGG16 Transfer Learning) to classify skin lesion images as **Benign** or **Malignant**.

---

## 📸 Demo

> Upload a dermoscopy image → Get instant AI prediction with confidence score.

---

## 🧠 Model

- **Architecture:** VGG16 (Transfer Learning) + GlobalAveragePooling + Dense Head
- **Input Size:** 128 × 128 pixels
- **Output:** Softmax — 2 classes (`Benign`, `Malignant`)
- **Dataset:** Melanoma Cancer Dataset (Benign & Malignant images)
- **Weights File:** `skin_cancer_model.weights.h5`

---

## 📁 Project Structure

```
skin-cancer-detector/
├── main.py                        # Streamlit app
├── skin_cancer_model.weights.h5   # Trained model weights (download separately)
├── requirements.txt               # Python dependencies
├── .gitignore
└── README.md
```

---

## ⚙️ Setup & Run Locally

### 1. Clone the repository
```bash
git clone https://github.com/your-username/skin-cancer-detector.git
cd skin-cancer-detector
```

### 2. Create a virtual environment
```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Add the model weights file
Place `skin_cancer_model.weights.h5` in the root project folder.  
> ⚠️ This file is excluded from Git (too large). Download it separately or train using the provided notebook.

### 5. Run the app
```bash
streamlit run main.py
```

---

## 🛠 Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.12 | Programming Language |
| Streamlit | Web App Framework |
| TensorFlow / Keras | Deep Learning |
| VGG16 | Pretrained CNN (Transfer Learning) |
| NumPy | Array Operations |
| Pillow | Image Processing |
| scikit-learn | Label Encoding / Metrics |

---

## ⚕️ Disclaimer

> This application is built for **academic/educational purposes only**.  
> It is **not a medical diagnostic tool** and should not replace professional dermatological advice.

---

## 👨‍💻 Author

**Your Name**  
Final Year B.Tech / BCA — Computer Science  
Mumbai, India
