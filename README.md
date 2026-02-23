# 🎭 Emotion Recognition App

Web aplikacija za prepoznavanje emocija iz audio snimaka koristeći tri različita AI modela.

## 🧠 Modeli

| Model | Tip | Opis |
|---|---|---|
| **TransformerAudioEmotion** | 🎵 Audio | Fine-tuned Transformer na audio značajkama |
| **LstmAudioEmotion** | 🎵 Audio | Bidirekcijski LSTM s attention mehanizmom |
| **PULSE_emotion** | 📝 Tekst | Whisper (STT) + RoBERTa LoRA za analizu teksta |

## 🏷️ Emocije

`angry` · `calm` · `disgust` · `fearful` · `happy` · `neutral` · `sad` · `surprised`

PULSE dodatno prepoznaje: `joy` · `sadness` · `anger` · `love` · `surprise`

---

## 🚀 Pokretanje

### Backend

```bash
cd backend
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python -m uvicorn main:app --port 8000
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Aplikacija dostupna na: `http://localhost:5173`

---

## 🗂️ Struktura projekta

```
emotion-app/
├── backend/
│   ├── main.py
│   ├── models.py
│   ├── transformer_model.py
│   ├── lstm_model.py
│   └── requirements.txt
└── frontend/
    ├── src/
    │   ├── App.jsx
    │   ├── App.css
    │   ├── api.js
    │   └── components/
    │       ├── AudioRecorder.jsx
    │       └── Results.jsx
    └── vite.config.js
```

---

## 🔌 API Endpointi

| Metoda | Endpoint | Opis |
|---|---|---|
| `GET` | `/health` | Status servera |
| `POST` | `/api/predict` | Sva 3 modela odjednom |
| `POST` | `/api/predict/transformer` | Samo Transformer |
| `POST` | `/api/predict/lstm` | Samo LSTM |
| `POST` | `/api/predict/pulse` | Samo PULSE |

---

## ⚙️ Tehnologije

**Backend:** FastAPI · PyTorch · Hugging Face Transformers · librosa · soundfile · PEFT

**Frontend:** React · Vite · Web Audio API

---

## 📋 Napomene

- Audio se konvertira u **WAV 16kHz mono** direktno u browseru putem Web Audio API — bez potrebe za ffmpeg-om
- Modeli se učitavaju s **Hugging Face Hub** pri prvom pokretanju
- PULSE koristi Whisper za transkripciju govora u tekst, zatim RoBERTa LoRA za klasifikaciju emocija
- Za najbolje rezultate snimi **5+ sekundi jasnog govora**
