# Shabda AI — Local Dictation Assistant

> A fully offline, cross-platform, real-time speech-to-text dictation engine.
> Works flawlessly on **macOS (Intel + Apple Silicon)** and **Windows (CPU + NVIDIA GPU)**.
> Highly optimized for ultra-low latency typing across all operating systems.

---

## 🌟 Perfect Cross-Platform Version
This project has been heavily calibrated to work out of the box on both Mac and Windows without any code changes. 
- **macOS Latency Optimized:** Uses a persistent `osascript` process to enable instant, zero-latency dictation directly into any Mac app without the usual `subprocess` startup delays or threading crashes.
- **Windows Latency Optimized:** Uses `pynput` with native string buffering to type text instantly into Windows applications, removing the slow character-by-character artificial typing delays.

---

## 🛠️ How It Works (The Engine Pipeline)

Shabda AI is built as a highly modular pipeline capturing speech and translating it into keyboard strokes in any app:

1. **Audio Capture (`sounddevice`):** Captures high-fidelity 16kHz audio directly from your selected microphone.
2. **Voice Activity Detection (`Silero VAD`):** Instantly detects speech versus silence to separate continuous background noise from your actual voice.
3. **Rolling Buffer:** Safely accumulates spoken words until a natural pause is detected.
4. **Neural Inference (`Faster-Whisper`):** Processes the audio payload instantly, yielding highly accurate text. It uses CPU (int8) natively on Mac, and can leverage NVIDIA CUDA on Windows automatically.
5. **Transcript Manager:** Curates the raw transcribed chunks into a polished, coherent paragraph.
6. **Dictation Engine:** Automatically detects your OS (macOS or Windows) and dynamically injects the text via native OS typing commands into *whichever app currently has focus* (e.g., your browser, Word document, or code editor).
7. **Floating UI:** Provides a neat, draggable microphone overlay that stays on top of all windows without ever stealing your typing focus.

---

## 🚀 Detailed Installation & Setup Instructions

### 🍎 macOS Setup (Apple Silicon M1/M2/M3 & Intel)

1. **Create and Activate a Virtual Environment:**
   Open your terminal and run:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   *(Note: macOS comes with all the necessary system typing tools (`osascript`); no extra system libraries are needed!)*

2. **Grant Permissions (Crucial for Mac):**
   The first time you run Shabda AI, macOS will block it from typing onto your screen for security reasons. You MUST grant the following:
   - Go to **System Settings** → **Privacy & Security** → **Accessibility**
   - Click the `+` button and add your `Terminal` app, `iTerm`, or the specific `python` executable you are using.
   - Go to **System Settings** → **Privacy & Security** → **Automation**
   - Allow `System Events` under your terminal or Python executable.
   *(If you skip this, it will listen, but it won't type anything into your documents!)*

3. **Run the App:**
   ```bash
   python main.py
   ```

### 🪟 Windows Setup (CPU & NVIDIA GPU)

1. **Create and Activate a Virtual Environment:**
   Open Command Prompt or PowerShell and run:
   ```cmd
   python -m venv venv
   .\venv\Scripts\activate
   ```

2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   
3. **NVIDIA GPU Acceleration (CUDA) - Optional but highly recommended:**
   If you have an NVIDIA GPU, Shabda AI will automatically utilize CUDA via Faster-Whisper to run large models instantly. For this to work perfectly on Windows, make sure you install PyTorch with CUDA support. Run this command:
   ```bash
   pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```
   *(This ensures you have all the necessary NVIDIA GPU libraries available for the AI engine).*

3. **Permissions (Optional):**
   - Generally, Windows allows `pynput` to type into apps seamlessly.
   - However, if you are trying to dictate into an application running as Administrator (e.g., Task Manager or certain IDEs), you must also run the Terminal/Command Prompt running Shabda AI as Administrator.

4. **Run the App:**
   ```bash
   python main.py
   ```

---

## 🎯 How to Use It

Once you run `python main.py`, a small **circular floating microphone overlay** will appear on the bottom right of your screen. 

### 🖱️ Overlay Controls

| Gesture | Action |
|---|---|
| **Left Click** | Start/Stop Listening (Turns red when listening, green when processing). |
| **Double Click** | Toggle **Dictation Mode** on/off (A small "D" badge appears). |
| **Right Click** | Open the hidden **Settings Window** for advanced controls. |
| **Click & Drag**| Move the tiny microphone window anywhere on your screen. |

### 🎙️ Using Dictation Mode

1. **Enable Dictation:** Double-click the microphone so the "D" badge appears.
2. **Focus an Application:** Click on ANY text field in ANY application (Chrome, MS Word, VS Code, Notes, etc.).
3. **Start Talking:** Click the microphone once (turns red) and start talking.
4. **Watch it Type:** Stop talking for a split second, and Shabda AI will instantly type the generated text directly into the application you are currently focused on, with exactly zero typing delay!

> **Pro Tip:** Shabda AI only injects the NEW text since the last pause. It assumes an "append-only" flow.

---

## ⚙️ Advanced Settings (Settings Window)

Right-click the overlay to access the settings panel:

- **General:** Change the Neural Engine model (tinier models are faster but less accurate, `medium.en` is great for Apple Silicon, `large-v3` is incredible if you have an NVIDIA GPU).
- **Microphone:** Choose the correct physical microphone you want the engine to listen to.
- **Keywords:** Add a list of contextually relevant words! If you often code, add `PyTorch, FastAPI, asyncio`. The Engine will hot-reload them and dramatically prioritize spelling those words correctly!
- **Transcript:** A raw view of everything dictated in the session.

---

## 🧠 Neural Models auto-downloaded

Models download automatically to the `models/` folder on first use.

| Model | Size | Best For |
|---|---|---|
| `tiny.en` | ~100 MB | Fastest, extremely low memory |
| `base.en` | ~145 MB | Fast, decent baseline |
| `small.en` | ~480 MB | Balanced (best for typical CPU usage) |
| `medium.en` | ~1.5 GB | Highly accurate (Default) |
| `large-v3` | ~3.0 GB | Max accuracy (Best with NVIDIA GPU) |

---

## 📜 License
MIT — completely offline, fully private, zero cloud APIs, zero telemetry.
