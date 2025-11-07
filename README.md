# Offline Voice Assistant

A fully functional 100% offline voice assistant with multi-language support using:
- **Vosk** for speech recognition
- **Piper TTS** for text-to-speech
- **llama-cpp-python** for language model inference
- **PyAudio** for audio capture

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Simply run:
```bash
python main.py
```

### Voice Commands
- Speak naturally to ask questions or have conversations
- Say "exit", "quit", or "goodbye" to stop the assistant
- Say "clear" or "reset" to clear conversation history

## Configuration

Configuration is stored in `assistant_config.json` and can be changed by running:

```bash
python main.py --init
```

The setup wizard will ask you to:
1. **Choose language**: English or German (Deutsch)
2. **Choose model size**:
   - Small (7B, ~4GB) - Faster, less accurate
   - Large (13B, ~7GB) - Slower, more accurate

**All required models are downloaded automatically!** No manual downloads needed.

## Features

✓ 100% offline operation
✓ Multi-language support (English & German)
✓ Configurable model sizes (7B or 13B)
✓ Automatic model downloads
✓ GPU acceleration (auto-detected)
✓ Real-time speech recognition
✓ Natural language understanding via local LLM
✓ High-quality text-to-speech (Piper)
✓ Conversation history tracking
✓ No internet connection required after setup

## Troubleshooting

### PyAudio Installation Issues
On Linux:
```bash
sudo apt-get install portaudio19-dev python3-pyaudio
```

On macOS:
```bash
brew install portaudio
```

### No microphone detected
Check your audio input device:
```bash
python -c "import pyaudio; p=pyaudio.PyAudio(); [print(p.get_device_info_by_index(i)) for i in range(p.get_device_count())]"
```

### GPU Acceleration
GPU acceleration is automatically detected and enabled. For CUDA support, install llama-cpp-python with CUDA:
```bash
CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install llama-cpp-python --force-reinstall --no-cache-dir
```

The assistant will automatically use GPU if available (CUDA or ROCm).
