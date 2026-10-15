# LifeSync AI 📅🤖

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12-blue.svg" alt="Python 3.12" />
  <img src="https://img.shields.io/badge/FastAPI-0.109-blue.svg" alt="FastAPI" />
  <img src="https://img.shields.io/badge/LangGraph-0.1-orange.svg" alt="LangGraph" />
  <img src="https://img.shields.io/badge/Expo-SDK%2055-black.svg" alt="Expo SDK 55" />
  <img src="https://img.shields.io/badge/Firebase-Firestore-red.svg" alt="Firestore" />
  <img src="https://img.shields.io/badge/Build-Passing-brightgreen.svg" alt="Build" />
</p>

**LifeSync AI** is a state-of-the-art, agentic daily schedule optimizer and lifestyle assistant. Built on **LangGraph** (agentic state orchestration), **FastAPI** (streaming backend), and **React Native/Expo** (mobile client), it dynamically constructs balanced, optimal daily agendas for students and professionals alike.

The system automatically balances academic deadlines, professional tasks, and hobbies around your **non-negotiable daily habits** (like sleep, workout, and meditation).

---

## 📖 Table of Contents
- [🌟 Key Capabilities](#-key-capabilities)
- [📈 Agentic Workflows & State Graph](#-agentic-workflows--state-graph)
- [📂 Project Directory Structure](#-project-directory-structure)
- [🗄️ Firestore Database Schema](#%EF%B8%8F-firestore-database-schema)
- [🔌 REST API References](#-rest-api-references)
- [⚙️ Setup & Installation](#%EF%B8%8F-setup--installation)
- [📅 Core Scheduling Engine Rules](#-core-scheduling-engine-rules)

---

## 🌟 Key Capabilities

### 🧠 Intent Classification with Heuristic Fallback
- Classified via a primary Gemini model (or Groq backup) and automatically falls back to regex-based classification for test users.
- **Smart Query Routing**: Intelligently identifies retrieval queries (e.g., *"What is my Monday timetable?"*) and routes them to chat instead of overwriting schedules.

### 🖼️ Timetable Vision Extraction
- Users upload a photo of their college or work timetable (sent as `multipart/form-data` file URI to prevent low-end device OOM crashes).
- Gemini parses the image, generates structured JSON classes, and stores them securely.

### 📚 Syllabus & Calendar Crawling
- Input your college (e.g., *RVCE, CSE, 5th Sem, 2022 Scheme*).
- An asynchronous background scraper fetches the college academic calendar and parses syllabus PDFs directly from the VTU database to feed into your tasks list.

### 🛡️ Wellness & Habit Protection
- Non-negotiable routines (like Meditation at 6:00 AM) are locked and protected from task/class overlaps.

---

## 📈 Agentic Workflows & State Graph

LifeSync AI uses a state graph to manage chat sessions, parse user details, build calendars, and interrupt for user approval.

```mermaid
graph TD
    User([User Chat/Upload]) --> API[FastAPI Endpoint]
    API --> Classify[Classify Intent Node]
    
    Classify -- info_provided --> Save[Save User Info Node]
    Classify -- schedule_build --> Hydrate[Hydrate Context Node]
    Classify -- general --> GenChat[General Chat Node]
    
    Save --> Clarify[Check Clarification Node]
    GenChat --> Clarify
    
    Hydrate --> Build[Build Schedule Node]
    Build --> Clarify
    
    Clarify -- Gaps Found --> PromptQuestion[Ask Discovery Question] --> END([End Turn])
    Clarify -- No Gaps & schedule_build --> Propose[Propose Schedule Node]
    
    Propose -- HITL Interrupted --> AppUser([User Approval API])
    AppUser -- Approved --> Commit[Commit Schedule Node]
    Commit --> END
```

---

## 📂 Project Directory Structure

The repository is divided into a Python backend service and a TypeScript Expo mobile application:

```
├── .agents/                    # Specialist agent scripts, prompt instructions, and rules
├── backend/
│   ├── app/
│   │   ├── agents/             # LangGraph state nodes, router, and chat stream controllers
│   │   │   ├── discovery.py    # Onboarding gap identifier
│   │   │   ├── graph.py        # LangGraph StateGraph builder
│   │   │   ├── llm.py          # Schema-constrained LLM entry points (astructured / atext)
│   │   │   ├── nodes.py        # Graph computation nodes (Build, Commit, Hydrate)
│   │   │   ├── parser.py       # Entity & info parser (LLM + regex heuristic fallback)
│   │   │   ├── schemas.py      # Pydantic models for every structured LLM output
│   │   │   └── state.py        # AgentState TypedDict definitions
│   │   ├── api/                # FastAPI routing paths (auth, chat streams, image uploads)
│   │   ├── data/               # Local JSON mappings & college catalogs
│   │   ├── memory/             # Postgres memory savers & checkpointer configurations
│   │   ├── tasks/              # APScheduler background jobs (Nightly Learning, Scrapers)
│   │   ├── tools/              # Specialized agent actions (Syllabus downloader, TMDB movie locator)
│   │   └── main.py             # FastAPI App root configuration
│   ├── tests/                  # pytest suite (in-memory Firestore/LLM fakes, no credentials needed)
│   ├── requirements.txt        # Backend python dependencies
│   ├── requirements-dev.txt    # pytest + pytest-asyncio
│   ├── .env.example            # Copy to .env and fill in
│   └── firebase-service-account.json [EXCLUDED] # Credentials
├── mobile/
│   ├── app/                    # Expo Router structure (Tabs: Schedule, Profile, Academics)
│   ├── constants/              # Theme details and color tokens
│   ├── hooks/                  # Custom hooks (useAgentChat, useGeofence, useSchedule)
│   ├── services/               # Firebase listeners and Local Notification triggers
│   ├── package.json            # Node.js dependencies
│   └── app.json                # Expo config manifest
├── README.md                   # Visual project guide
└── .gitignore                  # Ignored credentials list
```

---

## 🗄️ Firestore Database Schema

The system stores configuration files, user profile information, and schedule logs in Google Cloud Firestore:

| Collection Path | Document ID | Description | Key Fields |
| :--- | :--- | :--- | :--- |
| `users/{uid}` | `uid` | Core user preferences & onboarding state | `completed_discovery`, `college`, `branch`, `semester` |
| `timetables/{uid}` | `uid` | Extracted timetable classes by weekday | `Monday: [{"time": "09:00", "subject": "DSA"}]` |
| `academic_events/{uid}/items` | Auto-generated | Upcoming CIEs, exams, and assignment due dates | `title`, `type (cie/see)`, `due_date`, `completed` |
| `personal_schedule/{uid}` | `uid` | Fixed personal blocks (meditation, workouts) | `fixed_blocks: [{"title": "Meditation", "time": "06:00"}]` |
| `schedules/{uid}/days/{date}`| `YYYY-MM-DD` | User-approved daily schedule blocks | `approved: true`, `blocks: [{"time": "09:00", "title": "Math"}]` |
| `notifications/{uid}/pending`| `YYYY-MM-DD` | Scheduled notification reminders for the day | `reminders: [{"time": "08:50", "body": "Math in 10m"}]` |

---

## 🔌 REST API References

> [!NOTE]
> Every endpoint requires `Authorization: Bearer <firebase-id-token>`. The uid in
> the token is authoritative; a `user_id` in the request body that disagrees with
> it is rejected with `403`. When Firebase credentials are absent (local dev),
> the backend runs in a documented fallback that accepts the body `user_id` so
> the Expo mock-auth demo keeps working — see `backend/app/api/auth.py`.

### 1. Unified Agent Stream
- **Path**: `POST /api/chat`
- **Content-Type**: `application/json`
- **Request Body**:
  ```json
  {
    "user_id": "user_123",
    "thread_id": "thread_abc",
    "message": "schedule my day for 2026-06-15"
  }
  ```
- **Returns**: `text/event-stream` returning text tokens, intermediate tool outputs, or interrupted proposed schedules.
- **Thread namespacing**: the thread is stored as `{user_id}:{thread_id}`, so two
  users sending the same constant `thread_id` never share state.

### 2. Timetable Image Extraction
- **Path**: `POST /api/upload/timetable`
- **Content-Type**: `multipart/form-data`
- **Payload**: `file` (URI binary), `user_id` (string).
- **Response**:
  ```json
  {
    "status": "success",
    "timetable": {
      "Monday": [{"time": "09:00", "subject": "DSA"}]
    }
  }
  ```

### 3. Approve Proposed Schedule
- **Path**: `POST /api/schedule/approve`
- **Content-Type**: `application/json`
- **Request Body**:
  ```json
  {
    "thread_id": "thread_abc",
    "feedback": { "approved": true, "edits": [] }
  }
  ```
- **Response**: Status `success` alongside details of saved Firestore schedules.

---

## ⚙️ Setup & Installation

> [!IMPORTANT]
> Make sure PostgreSQL is running locally and you have created a database instance before launching the backend memory store.

### 1. Backend Setup
```bash
# Navigate & initialize virtualenv
cd backend
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Create .env file and edit configuration keys
cp .env.example .env
```

Run the test suite (no Postgres, Firebase, or API keys required):
```bash
pip install -r requirements-dev.txt
pytest
```
Start the local server:
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### 2. Mobile Setup
```bash
# Navigate & install node modules
cd ../mobile
npm install

# Run Dev Server
npx expo start
```

---

## 📅 Core Scheduling Engine Rules

LifeSync AI uses a deterministic rule-based generator when building user schedules to maintain calendar compliance:

| Trigger Scenario | Schedule Outcome | Styling/Notes |
| :--- | :--- | :--- |
| **Habits (e.g. Meditation)** | Lock strictly to requested times (e.g., 6:00 AM) | Type: `health` (Non-negotiable) |
| **CIE Exam $\le$ 7 days** | Insert **2-hour** study block daily | Type: `study` (Notes: *CIE Prep*) |
| **CIE Exam $\le$ 3 days** | Insert **3-hour** study block daily | Type: `study` (Notes: *CIE Prep CRITICAL*) |
| **Lab Record due tomorrow** | Insert **1.5-hour** writing block this evening | Type: `work` (Notes: *Urgent*) |
| **VTU Holiday (e.g. Yoga Day)**| Clear all university classes & study expectations | Rest day block inserted |
| **Saturday Night** | Query TMDB API, recommend film at **21:30** | Type: `entertainment` |
