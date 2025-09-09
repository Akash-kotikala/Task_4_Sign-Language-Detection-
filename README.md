# Task_4_Sign-Language-Detection-
Sign Language Detection Train ML model to recognize selected sign language words. Operate only from 6 PM to 10 PM. Provide GUI supporting image upload and real-time video detection. Ensure proper interface for user interaction.

### Sign Language detection Model Link :: 


https://drive.google.com/file/d/1VLIrQ0LxE3YC3xP_hc6DRH6Vn7JVJESv/view?usp=drive_link



### Dataset Link::


https://drive.google.com/file/d/1iVLdqsazevaQW5Ftm7Ehins6i1DoaW6c/view?usp=drive_link

# ✋🤟 Sign Language Detector  

## 📌 Problem Statement  
Classify hand gesture images representing **sign language words** (e.g., *hello, bye, thankyou, congratulations*).  

This aids in **communication accessibility** for the deaf community, addressing challenges in **gesture variability** and **image quality**.  

---

## 📂 Dataset  
- **Source:** Custom dataset (4 classes).  
- **Preprocessing:**  
  - Images resized → `128x128`  
  - Normalized  
  - Labels encoded (one-hot)  
- **Classes:** 4 → `hello`, `bye`, `thankyou`, `congratulations`  
- **Size:** Small (~MBs)  

---

## 🛠 Methodology  

### 🔹 Data Loading & Preprocessing  
- Load images via **Keras**  
- Normalize inputs  
- Encode labels → one-hot  

### 🔹 Model Architecture (CNN)  
- **Input:** `128x128x3`  
- **Layers:**  
  - 3️⃣ Conv layers + ReLU  
  - 🌀 Max-Pooling  
  - 🔒 Dropout  
  - 🔗 Dense layers  
- **Output:** Softmax (4 classes)  
- **Optimizer:** Adam (`lr=0.001`)  
- **Loss:** Categorical Crossentropy  
- **Metrics:** Accuracy  

### 🔹 Training  
- Train/Test Split → **80/20**  
- Epochs → **10**  
- Batch Size → **32**  

### 🔹 Evaluation  
- ✅ Accuracy (~95%)  
- ✅ Sample predictions (e.g., "thankyou")  
- ✅ Precision/Recall (high across classes)  

---

## ⚙ Tools & Libraries  
- 🧠 TensorFlow/Keras  
- 👁 OpenCV  
- 📊 Pandas  
- 📈 Matplotlib  
- 🔢 Scikit-learn  

---

## 📊 Results  
- Accuracy: ~95%  
- Sample Output: Correctly predicts gestures (e.g., "thankyou").  
- Limitation: Dataset is small → can be improved with augmentation & more classes.  

---

## 🚀 Installation  
```bash
pip install tensorflow opencv-python pandas scikit-learn matplotlib
