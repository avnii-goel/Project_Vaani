<div align="center">

# ✦ Vaani ✦
**Voice-first AI assistant for Indian government welfare schemes**

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![Next.js](https://img.shields.io/badge/Next.js-16-000000?style=flat-square&logo=nextdotjs&logoColor=white)](https://nextjs.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![LangGraph](https://img.shields.io/badge/LangGraph-agentic-FF6B35?style=flat-square)](https://langchain-ai.github.io/langgraph/)
[![Groq](https://img.shields.io/badge/Groq-Whisper_+_Llama4-F55036?style=flat-square)](https://groq.com)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

*Speak in Hindi. Get answers in Hindi. Apply for schemes in Hindi.*  
*No literacy required. No tech skills needed. Just your voice.*

</div>

---

## The Problem

Over 900 million Indians are eligible for government welfare schemes — yet most never claim their benefits. The reasons are consistent: language barriers, digital illiteracy, and the sheer complexity of navigating bureaucratic processes. A farmer in Bihar shouldn't need to read English to find out if he qualifies for PM-KISAN. A widow in Tamil Nadu shouldn't need a smartphone-savvy relative to apply for her pension.

**Vaani fixes this.**

---

## What Vaani Does

Vaani is a **voice-first AI assistant** that lets any citizen — regardless of language or digital experience — discover, understand, and apply for government welfare schemes through natural spoken conversation.

```
You speak  →  Vaani listens  →  Vaani searches  →  Vaani responds (in your language, in your voice)
```

1. **Speak** your question in Hindi, Bengali, Tamil, Telugu, Marathi, or English
2. **Groq Whisper** transcribes and translates to English
3. **LangGraph agent** (Llama 4 + Gemini 2.5) determines your eligibility and finds relevant schemes
4. **Gemini** generates a natural, spoken-style response
5. **Llama 4** translates it back to your language
6. **ElevenLabs** speaks the answer aloud in a natural voice

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        USER (Voice)                             │
└──────────────────────────┬──────────────────────────────────────┘
                           │  audio blob
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                   NEXT.JS FRONTEND  (:3000)                     │
│                                                                 │
│   /api/process-audio route (Next.js API)                        │
│   ├── Groq Whisper  →  transcription + English translation      │
│   ├── → FastAPI backend  →  AI response                         │
│   ├── Llama 4  →  translate response back to user's language    │
│   └── ElevenLabs  →  text-to-speech audio                       │
└──────────────────────────┬──────────────────────────────────────┘
                           │  POST /chat
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                  FASTAPI BACKEND  (:8000)                       │
│                                                                 │
│   LangGraph Agentic Workflow                                    │
│   ├── check_user_profile  →  SQLite                             │
│   ├── onboarding_step  →  collect name, age, state, income…     │
│   ├── supervisor  →  classify intent                            │
│   │   ├── welfare_search  →  Tavily + Gemini                    │
│   │   ├── form_generation  →  Tavily + Gemini HTML form         │
│   │   └── general_query  →  Gemini                              │
│   └── structure_final_response  →  Gemini                       │
└─────────────────────────────────────────────────────────────────┘
```

---

## Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| **Frontend** | Next.js 16 + TypeScript | Voice UI, language selector, conversation display |
| **Styling** | Tailwind CSS + Brutalist CSS | Bold, accessible design system |
| **Backend** | FastAPI + Python | REST API, request orchestration |
| **Agent** | LangGraph | Stateful multi-step agentic workflow |
| **STT** | Groq Whisper Large v3 | Speech-to-text + auto translation |
| **AI (fast)** | Groq Llama 4 Scout | Intent classification, data extraction, translation |
| **AI (rich)** | Gemini 2.5 Flash | Welfare search synthesis, form generation |
| **Search** | Tavily Search API | Real-time government scheme web search |
| **TTS** | ElevenLabs Multilingual v2 | Natural voice responses |
| **Database** | SQLite | User profiles, conversation history |

---

## Key Features

- **Voice-First User Experience:** Speak queries in your native language for a seamless experience.
- **Multilingual Support:** Handles multiple Indian languages and English.
- **AI-Powered Query Resolution:** Uses generative AI to understand complex questions and provide accurate responses.
- **End-to-End Automation:** Automates the entire workflow from voice input to actionable output.
- **Mobile Responsive:** Designed for accessibility on smartphones and tablets.
- **Privacy First:** User audio is processed securely and not stored beyond session needs.

---

## Project Structure

```
vaani/
├── backend/
│   ├── backend.py          # FastAPI app + LangGraph workflow
│   ├── requirements.txt    # Python dependencies
│   └── .env.example        # Environment variable template
│
├── frontend/
│   ├── src/
│   │   └── app/
│   │       ├── page.tsx                    # Main voice UI
│   │       ├── layout.tsx                  # Root layout + fonts
│   │       ├── globals.css                 # Brutalist design system
│   │       ├── api/
│   │       │   └── process-audio/
│   │       │       └── route.ts            # Audio pipeline (Whisper → Backend → ElevenLabs)
│   │       └── components/
│   │           ├── FormModal.tsx           # Scheme application form viewer
│   │           └── useAudioRecorder.ts     # MediaRecorder hook
│   ├── package.json
│   ├── next.config.ts
│   └── .env.local.example  # Frontend environment template
│
├── .gitignore
└── README.md
```

---

## Getting Started

### Prerequisites

- Python 3.10+
- Node.js 18+
- API keys for: Groq, Google Gemini, Tavily, ElevenLabs

### 1. Clone the repo

```bash
git clone https://github.com/yourusername/vaani.git
cd vaani
```

### 2. Backend setup

```bash
cd backend

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# → Fill in your API keys in .env

# Start the backend
python3 backend.py
# → Running at http://localhost:8000
# → API docs at http://localhost:8000/docs
```

### 3. Frontend setup

```bash
cd frontend

# Install dependencies
npm install

# Configure environment
cp .env.local.example .env.local
# → Fill in your API keys in .env.local

# Start the frontend
npm run dev
# → Running at http://localhost:3000
```

### 4. Open the app

Navigate to **http://localhost:3000** — select your language, click the mic, and start speaking.

---

## Environment Variables

### `backend/.env`

```env
GROQ_API_KEY=your_groq_api_key
GOOGLE_API_KEY=your_google_gemini_api_key
TAVILY_API_KEY=your_tavily_api_key
```

### `frontend/.env.local`

```env
GROQ_API_KEY=your_groq_api_key
GOOGLE_API_KEY=your_google_gemini_api_key
TAVILY_API_KEY=your_tavily_api_key
ELEVENLABS_API_KEY=your_elevenlabs_api_key
BACKEND_URL=http://127.0.0.1:8000/chat
NEXT_PUBLIC_BACKEND_BASE=http://127.0.0.1:8000
```

> All keys are free-tier available. Get them at [groq.com](https://groq.com) · [aistudio.google.com](https://aistudio.google.com) · [tavily.com](https://tavily.com) · [elevenlabs.io](https://elevenlabs.io)

---

## Supported Languages

- Hindi
- English
- (Extendable to other major Indian languages)

---

## Built With

- [Groq](https://groq.com/) — Whisper transcription + Llama 4 inference
- [Google Gemini](https://aistudio.google.com/) — Rich generation and form synthesis
- [LangGraph](https://langchain-ai.github.io/langgraph/) — Stateful agentic workflow
- [Tavily](https://tavily.com/) — Real-time web search
- [ElevenLabs](https://elevenlabs.io/) — Multilingual voice synthesis
- Government of India — Welfare scheme data sources

---

## Team

Built at "**[Hack-A-Tone (GydeXP & ARIES, IIT Delhi)]**" Hackathon by **Team KryptoKnights**

---

<div align="center">

*"Technology should work for everyone — not just those who can afford digital literacy."*

**⭐ Star this repo if Vaani inspired you**

</div>


## 📄 License

Distributed under the **MIT License**. See `LICENSE` for details.

---

