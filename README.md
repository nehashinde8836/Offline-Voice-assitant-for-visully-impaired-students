# 🎓 Offline Marathi Voice-Based AI Tutor for Visually Impaired Students

An AI-powered offline voice assistant designed specifically for visually impaired students to learn mathematics interactively in Marathi.  
The system supports speech recognition, voice responses, arithmetic problem solving, Pythagorean theorem explanations, quizzes, and story-based learning — all without requiring an internet connection.

The project focuses on accessibility, offline learning, and voice-driven education using lightweight AI and speech technologies.

---

# Features

- Offline Speech Recognition using Vosk
- Offline Text-to-Speech using pyttsx3
- Arithmetic Problem Solving
- Pythagorean Theorem Learning Module
- Interactive Voice-Based Quizzes
- Story-Based Mathematical Explanations
- Marathi Language Support
- Web Interface + Microphone Mode
- Accessibility-Focused Design for Visually Impaired Students
- Lightweight & Fully Offline System

---

# Tech Stack

## Programming Language
- Python

## Speech & Voice Technologies
- Vosk (Offline Speech Recognition)
- pyttsx3 (Offline Text-to-Speech)
- PyAudio

## Backend
- Flask

## Frontend
- HTML
- CSS
- JavaScript

## AI / Logic
- Rule-Based Intent Detection
- Dataset Retrieval System
- Arithmetic Engine
- Dynamic Quiz Generator

---

# Project Structure

```bash
PEC Project-3/
│
├── project/
│   ├── backend/
│   │   ├── app.py
│   │   ├── server.py
│   │   ├── math_engine.py
│   │   ├── intent.py
│   │   ├── quiz.py
│   │   ├── voice.py
│   │   ├── dataset_loader.py
│   │   └── llm_model/
│   │       ├── finetune.py
│   │       └── llm_inference.py
│   │
│   ├── dataset/
│   │   ├── marathi_math_dataset.jsonl
│   │   └── pythagoras_dataset.jsonl
│   │
│   ├── frontend/
│   │   └── index.html
│   │
│   └── vosk_model/
│
├── requirements.txt
├── README.md
│
├── test_core.py
├── test_bugs.py
├── test_bugfix.py
├── test_modes.py
├── test_pythagoras.py
├── test_quiz_fix.py
├── test_session.py
├── test_updated_dataset.py
│
├── generate_dataset.py
├── generate_pythagoras_dataset.py
├── generate_rich_pythagoras_dataset.py
│
├── marathi_math_dataset.jsonl
└── pythagoras_dataset.jsonl
```

---

# ⚙️ Installation & Setup

## 1️⃣ Clone the Repository

```bash
git clone https://github.com/nehashinde8836/Offline-Voice-assitant-for-visully-impaired-students.git
```

---

## 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### For Windows Users (PyAudio Fix)

```bash
pip install pipwin
pipwin install pyaudio
```

---

## 3️⃣ Download Vosk Model

Download the Hindi Vosk model from:

https://alphacephei.com/vosk/models

Recommended model:

```bash
vosk-model-small-hi-0.22
```

Extract the downloaded folder inside:

```bash
project/vosk_model/
```

---

# ▶️ Running the Project

## Option A — Web Interface (Recommended)

```bash
python project/backend/server.py
```

Open browser:

```bash
http://localhost:5000
```

---

## Option B — Voice Assistant Mode

```bash
python project/backend/app.py
```

---

## Option C — Run Test Files

```bash
python test_core.py
python test_pythagoras.py
python test_bugs.py
```

---

# Supported Voice Commands

| Marathi Command | Function |
|---|---|
| `पाच अधिक तीन` | Arithmetic calculation |
| `10 वजा 4` | Subtraction |
| `पायथागोरस समजाव` | Pythagorean explanation |
| `त्रिकोण गोष्ट सांग` | Story-based explanation |
| `कर्ण उदाहरण दाखव` | Pythagoras example |
| `क्विझ सुरू कर` | Start quiz |
| `परत सांगा` | Repeat last response |
| `बंद करा` | Exit application |

---

# System Architecture

```text
Voice Input (Marathi)
        ↓
Vosk Speech Recognition
        ↓
Intent Detection Engine
        ↓
┌──────────────────────────────┐
│ Arithmetic Solver            │
│ Pythagoras Formula Engine    │
│ Dataset Retrieval System     │
│ Quiz Generator               │
│ Story Explanation Module     │
└──────────────────────────────┘
        ↓
pyttsx3 Text-to-Speech
        ↓
Voice Output in Marathi
```

---

# Accessibility Features

- Fully voice-controlled interaction
- Offline learning support
- Large microphone interaction support
- Audio-based mathematical explanations
- Designed for visually impaired students
- Lightweight CPU-friendly architecture

---

# Future Enhancements

- OCR-based Text Reading
- Multi-language Support
- AI Chatbot Integration
- Mobile Application Version
- Smart Educational Recommendation System
- Advanced Voice Personalization

---

# Author

Neha Shinde

- GitHub: https://github.com/nehashinde8836/Offline-Voice-assitant-for-visully-impaired-students
- LinkedIn: https://www.linkedin.com/in/neha-shinde-software-engineer/

---

# 📄 License

This project is developed for educational and accessibility purposes.
