# AI SIMD Compiler Agent Orchestrator

This is a FastAPI-based orchestrator that uses Gemini to generate C++ SIMD (AVX2) code, compiles it using `g++`, and benchmarks it against a scalar implementation.

## Features
- **Prompt-to-SIMD:** Converts natural language descriptions of mathematical operations into optimized AVX2 code.
- **Automated Benchmarking:** Compiles and runs a high-resolution benchmark to measure performance gains.
- **Safety Checks:** Verifies that the SIMD results match the scalar results.

## Prerequisites
- Arch Linux (or any Linux with `g++` and AVX2 support)
- Python 3.10+
- Google Gemini API Key

## Setup

1. Install system dependencies:
   ```bash
   sudo pacman -S gcc
   ```

2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set your Gemini API key:
   ```bash
   export GENAI_API_KEY='your_api_key_here'
   ```

## Running the App

```bash
python app.py
```

The server will start at `http://0.0.0.0:8000`.

## Example Usage

Send a `POST` request to `/optimize`:

```bash
curl -X POST http://localhost:8000/optimize \
     -H "Content-Type: application/json" \
     -d '{"prompt": "Vector addition of two float arrays"}'
```

## Response Format
```json
{
  "scalar_code": "...",
  "simd_code": "...",
  "speedup": 4.2,
  "scalar_time_ms": 12.5,
  "simd_time_ms": 2.9,
  "compilation_output": ""
}
```
