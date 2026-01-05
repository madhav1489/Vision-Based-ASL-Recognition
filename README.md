# Vision-Based American Sign Language Recognition System

This project converts **American Sign Language (ASL)** hand gestures into **text and speech** using real-time **computer vision and machine learning**.  
It is designed as an assistive communication tool for hearing-impaired users.

---

## 🚀 Features
- Real-time ASL hand gesture recognition using webcam
- Hand landmark detection and skeletal visualization
- ASL alphabet to text conversion
- Sentence formation with intelligent word suggestions
- Text-to-speech output
- Interactive GUI built using Tkinter

---

## 🛠️ Tech Stack
- **Programming Language:** Python  
- **Computer Vision:** OpenCV  
- **Machine Learning:** Scikit-learn  
- **GUI:** Tkinter  
- **Libraries:** NumPy, PIL, Joblib  

---

## 📁 Project Structure
ASL/
│── app.py
│── requirements.txt
│
├── hand_tracking/
│ └── detector.py
│
├── ml/
│ ├── generate_data.py
│ ├── train_svm.py
│ └── delete_letter.py
│
├── utils/
│ ├── sentence_builder.py
│ └── words.txt
│
├── speech/
│ └── tts.py
│
└── models/
└── (trained model file – ignored in GitHub)

--
Install dependencies
pip install -r requirements.txt

3️⃣ Run the application
python app.py
⚠️ Note: Trained model files are excluded from the repository due to size constraints.


📸This Shows how the data was Captured:


<img width="1919" height="1013" alt="Screenshot 2026-01-05 004241" src="https://github.com/user-attachments/assets/accbecc9-367d-481a-9e67-84ebe5c0e693" />

📸the trained model with suggestion and voice

<img width="1919" height="1079" alt="Screenshot 2026-01-05 110258" src="https://github.com/user-attachments/assets/e7d42628-dfa9-4f8c-8440-419879fbbfbf" />




