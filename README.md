# Shabda AI — Core Speech-To-Text Engine

An ultra-low latency, exceptionally accurate, cross-platform, strictly offline Speech-To-Text engine.

Engineered natively with heavy tensor-quantization, dynamic sliding-window audio buffering, and zero-hallucination guardrails, this project is built to perfectly mimic the "Live Dictation" features of MS Word and Google Translate using local AI.

---

## 🚀 Features

- **Zero API Keys Required:** Runs entirely locally/offline on your machine keeping your data natively secure.
- **Cross-Platform Auto-Detection:** Automatically optimizes for Mac (Apple Silicon) or Windows (NVIDIA CUDA / CPU).
- **Interactive UI Menu:** Launch effortlessly without relying on complex terminal flags.
- **Live Streaming Dictation:** Your words magically appear and update dynamically natively as you speak, seamlessly flowing into structured paragraphs.
- **Keyword Correction Engine:** Pre-loaded with RapidFuzz mathematically scanning for specific jargon (e.g., PySpark, Databricks). Any structural mistakes the AI makes on domain-specific vocabulary are instantly fixed.
- **Hallucination Prevention:** Hardcoded logic filters and context locks explicitly ban the AI from guessing or writing ghost words or foreign languages when the room is silent.

---

## 🛠️ Installation & Setup

### For Mac (Apple Silicon: M1 / M2 / M3)
Mac systems will strictly bypass the GPU and use your CPU with `int8` (8-bit mathematically quantized) acceleration natively optimizing matrix math without draining battery levels.

1. **Clone the project and open the terminal inside the folder.**
2. **Setup a virtual environment (Recommended):**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

### For Windows (NVIDIA GPU Recommended)
Windows natively supports CUDA graphic card tensor mapping. Because of this, Windows users can mathematically achieve a staggering 10x-15x transcription speed increase by routing audio logic directly into their Graphics Card!

1. **Clone the project and open PowerShell / Command Prompt inside the folder.**
2. **Setup a virtual environment:**
   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```
3. **Install Main Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
4. **Link Python to your NVIDIA Driver (Crucial for GPU Acceleration):**
   Run the following command to securely bind the underlying Neural Networks directly into Windows CUDA cores.
   ```bash
   pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```
   *(If you do not have an NVIDIA GPU, skip step 4. Shabda AI will automatically self-recognize this and fall back to running securely on your CPU).*

---

## 💻 How To Run

You no longer need to edit any Python code manually! Simply execute the application directly:

```bash
python main.py
```

Upon launching, a beautifully rendered **Interactive Menu** will appear asking you to specify your engine's physical size restraint. Type a number `[1-5]` and hit enter!

> **Note on Initial Boot:** The extremely first time you select a model size, the engine will halt momentarily to securely download its neural weights into your hidden cache. *Subsequent launches skip this entirely and boot instantly.*

### Understanding the Model Array:
- `[1]` **tiny.en (~100MB)**: Utterly instantaneous latency. Recommended strictly for highly power-constrained generic laptops.
- `[2]` **base.en (~145MB)**: Strong minimal-tier transcription baseline.
- `[3]` **small.en (~480MB)**: **The Goldilocks Model**. Perfectly balances native CPU hardware throughput with heavily reinforced grammar tracking. Highly recommended for Mac architecture.
- `[4]` **medium.en (~1.5GB)**: Exceptional native English accuracy. This model physically demands a highly capable modern processor or ideally an Nvidia GPU.
- `[5]` **large-v3 (~3.0GB)**: Flawless human-tier studio grammar level. Because of its sheer monolithic file weight, this is strictly recommended for Windows systems equipped with discrete NVIDIA Graphics Cards. 

*(For advanced automation logic, you can easily bypass the menu natively by triggering the flags via terminal):*
```bash
python main.py --model small.en
```

---

## 📝 Editing the Keywords Dictionary
If Shabda AI fails to transcribe highly complex industry vocabulary correctly (e.g. `Databricks`, `Z-Ordering`, etc) because it is outside general baseline knowledge, simply open `keywords.json` in any text editor. 

You can freely append as many thousands of required words into the JSON array as you wish. 

The next time you boot `main.py`, the RapidFuzz engine will immediately load and perfectly memorize them!

---

## 🧠 Under The Hood (Architecture)

If you are a developer intending to modify Shabda AI, here is a functional overview of the pipeline workflow logic:

1. **`audio_engine.py`:** Seamlessly captures native audio chunks directly avoiding OS level bottlenecks using `sounddevice`.
2. **`vad_engine.py` (Voice Activity Detector):** A proprietary sub-model (Silero) constantly scanning your microphone locally in 32ms frames. It explicitly drops dead static and routes only genuine human frequencies mathematically over a `0.3` probability-index.
3. **`buffer_engine.py`:** An asynchronous mechanism conceptually combining those scattered voice frames into continually overlapping sentence arrays. This stops the Whisper AI from brutally severing your sentences mid-word.
4. **`inference_engine.py`:** Natively binds to CTranslate2 Faster-Whisper. Dynamically overrides and crushes *hallucinations* (forcing greedy translation math and explicitly preventing YouTube-subtitling loop leaks). 
5. **`post_engine.py`:** Executes local Regex formatting limits and uses RapidFuzz mathematically to intelligently assess specific technical domain language distances from your `keywords.json` array.
6. **`main.py`:** Renders the CLI user experience, mapping dynamic carriage returns and `rich.Live` panels to make the output update intelligently based on live asynchronous context queues.
