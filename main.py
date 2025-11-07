#!/usr/bin/env python3
"""
Fully Offline Voice Assistant
Uses Vosk (speech recognition), pyttsx3 (TTS), and llama-cpp-python (LLM)
"""

import os
import sys
import json
import queue
import zipfile
import urllib.request
import warnings
import argparse
from pathlib import Path

# Suppress ALSA/JACK warnings
os.environ['ALSA_CARD'] = 'default'
if sys.platform == 'linux':
    # Redirect stderr temporarily to suppress ALSA warnings
    stderr = sys.stderr
    sys.stderr = open(os.devnull, 'w')

import pyaudio

if sys.platform == 'linux':
    # Restore stderr
    sys.stderr.close()
    sys.stderr = stderr

from vosk import Model, KaldiRecognizer
from llama_cpp import Llama
import wave
import subprocess

# Suppress llama-cpp warnings
warnings.filterwarnings('ignore', category=RuntimeWarning, module='llama_cpp')

# Configuration file - stored in data/config directory
CONFIG_DIR = Path("data/config")
CONFIG_FILE = CONFIG_DIR / "assistant_config.json"

# Model configurations
MODEL_CONFIGS = {
    "llama": {
        "small": {
            "name": "llama-2-7b-chat.Q4_K_M.gguf",
            "url": "https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q4_K_M.gguf",
            "size": "~4GB"
        },
        "large": {
            "name": "llama-2-13b-chat.Q4_K_M.gguf",
            "url": "https://huggingface.co/TheBloke/Llama-2-13B-Chat-GGUF/resolve/main/llama-2-13b-chat.Q4_K_M.gguf",
            "size": "~7GB"
        }
    },
    "vosk": {
        "en": {
            "name": "vosk-model-small-en-us-0.15",
            "url": "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip"
        },
        "de": {
            "name": "vosk-model-small-de-0.15",
            "url": "https://alphacephei.com/vosk/models/vosk-model-small-de-0.15.zip"
        }
    },
    "piper": {
        "en": {
            "model": "en_US-lessac-medium.onnx",
            "config": "en_US-lessac-medium.onnx.json",
            "url_base": "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/lessac/medium"
        },
        "de": {
            "model": "de_DE-thorsten-medium.onnx",
            "config": "de_DE-thorsten-medium.onnx.json",
            "url_base": "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/de/de_DE/thorsten/medium"
        }
    }
}

SAMPLE_RATE = 16000
CHUNK_SIZE = 4096

# Default configuration
DEFAULT_CONFIG = {
    "language": "en",
    "llama_model": "small",
    "initialized": False
}

# Multilingual messages
MESSAGES = {
    "en": {
        "greeting": "Hello! I am your offline voice assistant. How can I help you today?",
        "goodbye": "Goodbye! Have a great day!",
        "cleared": "Conversation history cleared. What would you like to talk about?",
        "no_speech": "No speech detected. Please try again."
    },
    "de": {
        "greeting": "Hallo! Ich bin dein Offline-Sprachassistent. Wie kann ich dir heute helfen?",
        "goodbye": "Auf Wiedersehen! Hab einen schönen Tag!",
        "cleared": "Gesprächsverlauf gelöscht. Worüber möchtest du sprechen?",
        "no_speech": "Keine Sprache erkannt. Bitte versuche es erneut."
    }
}

def load_config():
    """Load configuration from file or return default"""
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE, 'r') as f:
            return json.load(f)
    return DEFAULT_CONFIG.copy()

def save_config(config):
    """Save configuration to file"""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_FILE, 'w') as f:
        json.dump(config, f, indent=2)

def get_model_paths(config):
    """Get model paths based on configuration"""
    lang = config["language"]
    llama_size = config["llama_model"]

    # Models directory - inside data directory
    models_dir = Path("data/models")

    # Vosk paths - now in models directory
    vosk_config = MODEL_CONFIGS["vosk"][lang]
    vosk_model_name = vosk_config["name"]
    vosk_model_path = models_dir / vosk_model_name
    vosk_model_url = vosk_config["url"]

    # Llama paths
    llama_config = MODEL_CONFIGS["llama"][llama_size]
    llama_model_name = llama_config["name"]
    llama_model_path = models_dir / llama_model_name
    llama_model_url = llama_config["url"]

    # Piper paths
    piper_config = MODEL_CONFIGS["piper"][lang]
    piper_model_name = piper_config["model"]
    piper_config_name = piper_config["config"]
    piper_model_path = models_dir / piper_model_name
    piper_config_path = models_dir / piper_config_name
    piper_model_url = f"{piper_config['url_base']}/{piper_model_name}"
    piper_config_url = f"{piper_config['url_base']}/{piper_config_name}"

    return {
        "vosk_model_path": str(vosk_model_path),
        "vosk_model_url": vosk_model_url,
        "vosk_model_name": vosk_model_name,
        "llama_model_path": str(llama_model_path),
        "llama_model_url": llama_model_url,
        "llama_model_name": llama_model_name,
        "piper_model_path": str(piper_model_path),
        "piper_config_path": str(piper_config_path),
        "piper_model_url": piper_model_url,
        "piper_config_url": piper_config_url,
        "piper_model_name": piper_model_name,
        "piper_config_name": piper_config_name
    }

def detect_gpu():
    """Detect if CUDA/GPU is available"""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"GPU detected: {torch.cuda.get_device_name(0)}")
            return True
    except ImportError:
        pass

    # Check for ROCm (AMD)
    try:
        result = subprocess.run(["rocm-smi"], capture_output=True, timeout=2)
        if result.returncode == 0:
            print("AMD GPU detected (ROCm)")
            return True
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
        pass

    print("No GPU detected, using CPU")
    return False

def download_with_progress(url, destination):
    """Download file with progress bar"""
    def report_progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        percent = min(downloaded * 100 / total_size, 100)
        bar_length = 40
        filled = int(bar_length * percent / 100)
        bar = '█' * filled + '░' * (bar_length - filled)
        print(f'\r[{bar}] {percent:.1f}% ({downloaded / (1024*1024):.1f}/{total_size / (1024*1024):.1f} MB)', end='', flush=True)

    print(f"Downloading from {url}...")
    urllib.request.urlretrieve(url, destination, reporthook=report_progress)
    print()  # New line after download

def download_vosk_model(model_path, model_url, model_name):
    """Download and extract Vosk speech recognition model"""
    model_path_obj = Path(model_path)
    if model_path_obj.exists():
        print(f"Vosk model already exists at {model_path}")
        return

    print("Vosk model not found. Downloading...")
    models_dir = Path("data/models")
    models_dir.mkdir(parents=True, exist_ok=True)
    zip_path = models_dir / f"{model_name}.zip"

    try:
        download_with_progress(model_url, str(zip_path))

        print("Extracting model...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(str(models_dir))

        zip_path.unlink()
        print(f"Vosk model downloaded and extracted to {model_path}")
    except Exception as e:
        print(f"Error downloading Vosk model: {e}")
        print("Please download manually from: https://alphacephei.com/vosk/models")
        sys.exit(1)

def download_llama_model(model_path, model_url):
    """Download LLM model"""
    model_path_obj = Path(model_path)
    if model_path_obj.exists():
        print(f"LLM model already exists at {model_path}")
        return

    print("LLM model not found. Downloading...")
    models_dir = Path("data/models")
    models_dir.mkdir(parents=True, exist_ok=True)

    try:
        download_with_progress(model_url, model_path)
        print(f"LLM model downloaded to {model_path}")
    except Exception as e:
        print(f"Error downloading LLM model: {e}")
        print("\nAlternative models you can download manually:")
        print("  - Smaller (faster): https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF")
        print("  - Larger (better): https://huggingface.co/TheBloke/Llama-2-13B-Chat-GGUF")
        sys.exit(1)

def download_piper_model(model_path, config_path, model_url, config_url):
    """Download Piper TTS model and config"""
    models_dir = Path("data/models")
    models_dir.mkdir(parents=True, exist_ok=True)

    model_path_obj = Path(model_path)
    config_path_obj = Path(config_path)

    # Download model file
    if not model_path_obj.exists():
        print("Piper TTS model not found. Downloading...")
        try:
            download_with_progress(model_url, model_path)
            print(f"Piper model downloaded to {model_path}")
        except Exception as e:
            print(f"Error downloading Piper model: {e}")
            sys.exit(1)
    else:
        print(f"Piper model already exists at {model_path}")

    # Download config file
    if not config_path_obj.exists():
        print("Downloading Piper config...")
        try:
            download_with_progress(config_url, config_path)
            print(f"Piper config downloaded to {config_path}")
        except Exception as e:
            print(f"Error downloading Piper config: {e}")
            sys.exit(1)
    else:
        print(f"Piper config already exists at {config_path}")

def interactive_setup():
    """Interactive setup wizard"""
    print("=" * 60)
    print("VOICE ASSISTANT SETUP")
    print("=" * 60)
    print()

    config = DEFAULT_CONFIG.copy()

    # Language selection
    print("Select language:")
    print("  1. English")
    print("  2. German (Deutsch)")
    lang_choice = input("Enter choice (1-2) [1]: ").strip() or "1"

    if lang_choice == "2":
        config["language"] = "de"
        print("Language set to: German\n")
    else:
        config["language"] = "en"
        print("Language set to: English\n")

    # Model size selection
    print("Select LLM model size:")
    print("  1. Small (7B, ~4GB) - Faster, less accurate")
    print("  2. Large (13B, ~7GB) - Slower, more accurate")
    model_choice = input("Enter choice (1-2) [1]: ").strip() or "1"

    if model_choice == "2":
        config["llama_model"] = "large"
        print("Model set to: Large (13B)\n")
    else:
        config["llama_model"] = "small"
        print("Model set to: Small (7B)\n")

    config["initialized"] = True
    save_config(config)

    print("Configuration saved!")
    print("\nModels will be downloaded automatically on first run.")
    print(f"Starting assistant with: {config['language'].upper()} | {config['llama_model'].upper()} model\n")

    return config

class OfflineVoiceAssistant:
    def __init__(self, config):
        print("Initializing Offline Voice Assistant...")

        # Get model paths based on config
        self.config = config
        self.paths = get_model_paths(config)

        # Download models if needed
        download_vosk_model(
            self.paths["vosk_model_path"],
            self.paths["vosk_model_url"],
            self.paths["vosk_model_name"]
        )
        download_llama_model(
            self.paths["llama_model_path"],
            self.paths["llama_model_url"]
        )
        download_piper_model(
            self.paths["piper_model_path"],
            self.paths["piper_config_path"],
            self.paths["piper_model_url"],
            self.paths["piper_config_url"]
        )

        # Initialize speech recognition
        print("Loading speech recognition model...")
        self.vosk_model = Model(self.paths["vosk_model_path"])
        self.recognizer = KaldiRecognizer(self.vosk_model, SAMPLE_RATE)

        # Initialize text-to-speech (Piper TTS)
        print("Initializing text-to-speech with Piper TTS...")
        self.piper_model_path = self.paths["piper_model_path"]
        # Check if piper command is available
        try:
            result = subprocess.run(["piper", "--version"], capture_output=True, timeout=5)
            if result.returncode != 0:
                print("Warning: Piper TTS command not found. Speech output will be disabled.")
        except (subprocess.TimeoutExpired, FileNotFoundError):
            print("Warning: Piper TTS not available. Speech output will be disabled.")

        # Initialize LLM with GPU support if available
        print("Loading language model (this may take a moment)...")
        has_gpu = detect_gpu()

        # Determine optimal settings
        n_gpu_layers = 35 if has_gpu else 0  # Offload most layers to GPU if available
        n_threads = 4 if not has_gpu else 1  # Use more CPU threads only if no GPU

        print(f"LLM config: GPU layers={n_gpu_layers}, CPU threads={n_threads}")

        self.llm = Llama(
            model_path=self.paths["llama_model_path"],
            n_ctx=2048,
            n_threads=n_threads,
            n_gpu_layers=n_gpu_layers,
            use_mmap=True,
            use_mlock=False
        )

        # Initialize audio
        self.audio = pyaudio.PyAudio()
        self.audio_queue = queue.Queue()
        self.is_listening = False

        # Conversation history
        self.conversation_history = []

        print("Initialization complete!\n")

    def speak(self, text):
        """Convert text to speech using Piper TTS"""
        print(f"Assistant: {text}")

        try:
            # Use command-line piper for simplicity and reliability
            temp_audio = Path("data/temp_speech.wav")

            # Always use subprocess for Piper
            subprocess.run(
                ["piper", "--model", self.piper_model_path, "--output_file", str(temp_audio)],
                input=text.encode('utf-8'),
                check=True,
                capture_output=True
            )

            # Play the generated audio
            wf = wave.open(str(temp_audio), 'rb')

            stream = self.audio.open(
                format=self.audio.get_format_from_width(wf.getsampwidth()),
                channels=wf.getnchannels(),
                rate=wf.getframerate(),
                output=True
            )

            # Play audio
            data = wf.readframes(1024)
            while data:
                stream.write(data)
                data = wf.readframes(1024)

            # Cleanup
            stream.stop_stream()
            stream.close()
            wf.close()

            # Remove temporary file
            if temp_audio.exists():
                temp_audio.unlink()

        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            print(f"Error: Piper TTS not available or failed - {e}")
            print("Install piper: pip install piper-tts")
        except Exception as e:
            print(f"Error in speak method: {e}")
            import traceback
            traceback.print_exc()

    def listen(self):
        """Capture audio and convert to text"""
        print("Listening... (speak now)")

        stream = self.audio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=SAMPLE_RATE,
            input=True,
            frames_per_buffer=CHUNK_SIZE
        )

        stream.start_stream()

        text = ""
        silence_chunks = 0
        max_silence = 30  # Stop after ~2 seconds of silence

        try:
            while True:
                data = stream.read(CHUNK_SIZE, exception_on_overflow=False)

                if self.recognizer.AcceptWaveform(data):
                    result = json.loads(self.recognizer.Result())
                    text = result.get('text', '')
                    if text:
                        break
                else:
                    partial = json.loads(self.recognizer.PartialResult())
                    partial_text = partial.get('partial', '')

                    if partial_text:
                        silence_chunks = 0
                        print(f"\rRecognizing: {partial_text}", end='', flush=True)
                    else:
                        silence_chunks += 1
                        if silence_chunks > max_silence:
                            # Get final result
                            result = json.loads(self.recognizer.FinalResult())
                            text = result.get('text', '')
                            break

        except KeyboardInterrupt:
            pass
        finally:
            stream.stop_stream()
            stream.close()
            print()  # New line after recognition

        return text.strip()

    def generate_response(self, user_input):
        """Generate response using offline LLM"""
        # Add to conversation history
        self.conversation_history.append({"role": "user", "content": user_input})

        # Build prompt without leading <s> to avoid duplication
        prompt = ""
        for msg in self.conversation_history[-6:]:  # Keep last 3 exchanges
            if msg["role"] == "user":
                prompt += f"[INST] {msg['content']} [/INST] "
            else:
                prompt += msg["content"] + " "

        print("Generating response...")

        # Generate response
        output = self.llm(
            prompt.strip(),
            max_tokens=256,
            temperature=0.7,
            top_p=0.9,
            stop=["</s>", "[INST]", "[/INST]"],
            echo=False
        )

        response = output['choices'][0]['text'].strip()

        # Add to conversation history
        self.conversation_history.append({"role": "assistant", "content": response})

        return response

    def get_message(self, key):
        """Get message in the configured language"""
        lang = self.config["language"]
        return MESSAGES[lang][key]

    def process_command(self, text):
        """Process special commands"""
        text_lower = text.lower()

        # English commands
        exit_words_en = ['exit', 'quit', 'goodbye', 'bye']
        clear_words_en = ['clear', 'reset', 'new conversation']

        # German commands
        exit_words_de = ['beenden', 'tschüss', 'auf wiedersehen']
        clear_words_de = ['löschen', 'zurücksetzen', 'neue konversation']

        # Check for exit commands
        if any(word in text_lower for word in exit_words_en + exit_words_de):
            return "exit"
        # Check for clear commands
        elif any(word in text_lower for word in clear_words_en + clear_words_de):
            self.conversation_history = []
            return "cleared"
        else:
            return None

    def run(self):
        """Main assistant loop"""
        self.speak(self.get_message("greeting"))

        try:
            while True:
                # Listen for user input
                user_input = self.listen()

                if not user_input:
                    print(self.get_message("no_speech"))
                    continue

                print(f"You: {user_input}")

                # Process commands
                command = self.process_command(user_input)

                if command == "exit":
                    self.speak(self.get_message("goodbye"))
                    break
                elif command == "cleared":
                    self.speak(self.get_message("cleared"))
                    continue

                # Generate and speak response
                response = self.generate_response(user_input)
                self.speak(response)

        except KeyboardInterrupt:
            print("\nShutting down...")
            self.speak(self.get_message("goodbye"))

        finally:
            self.cleanup()

    def cleanup(self):
        """Clean up resources"""
        if hasattr(self, 'audio'):
            self.audio.terminate()
        print("Assistant shut down successfully.")

def main():
    """Entry point"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Offline Voice Assistant")
    parser.add_argument("--init", action="store_true", help="Run interactive setup")
    args = parser.parse_args()

    print("=" * 60)
    print("OFFLINE VOICE ASSISTANT")
    print("=" * 60)
    print()

    # Load or create configuration
    config = load_config()

    # Run setup if --init flag or not initialized
    if args.init or not config.get("initialized", False):
        config = interactive_setup()

    print(f"Configuration: Language={config['language'].upper()}, Model={config['llama_model'].upper()}")
    print()

    # Start assistant
    try:
        assistant = OfflineVoiceAssistant(config)
        assistant.run()
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        print("\nMake sure you have installed:")
        print("  pip install -r requirements.txt")
        print("\nRequired: vosk, pyaudio, piper-tts, llama-cpp-python")
        print("\nRun with --init to reconfigure the assistant.")
        sys.exit(1)

if __name__ == "__main__":
    main()
