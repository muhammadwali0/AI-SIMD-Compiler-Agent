# AI SIMD Compiler Agent
### Natural Language → Validated AVX2 Intrinsics

AI-powered agent that translates natural language descriptions of mathematical operations into validated, benchmarked AVX2 intrinsics. Compiles and stress-tests generated C++ against a true scalar reference, with an agentic retry loop on compiler errors and semantic mismatches.

Built for systems engineers who are tired of reading the Intel Intrinsics Guide.

---

## Demo

[![SIMD-AGENT Demo](https://img.youtube.com/vi/qNxoIv_bGLw/maxresdefault.jpg)](https://www.youtube.com/watch?v=qNxoIv_bGLw)

---

## The Problem

The Intel Intrinsics Guide has ~6000 entries. Writing AVX2 code for even a simple operation — a horizontal sum, a masked reduction, an FMA chain — requires knowing which intrinsics exist, how lane isolation breaks naive implementations, and how to compose multiple operations in the correct order. This is a manually intensive, documentation-heavy workflow that even experienced systems engineers spend significant time on.

SIMD-AGENT collapses that cycle into a single natural language prompt.

---

## How It Works

```
Natural Language Input
        │
        ▼
┌─────────────────────┐
│   Intent Parser     │  Gemini API — NL → structured operation schema
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│   Code Synthesizer  │  Gemini API — schema → AVX2 intrinsics + scalar reference
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│  Validation Engine  │  clang + 64K float test vectors
│  Compile → Run →    │  Diffs SIMD output against scalar within epsilon
│  Diff → Certify     │
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│  Agentic Retry Loop │  Compiler errors and semantic mismatches fed back
│  (up to 3 attempts) │  to Gemini for corrected generation
└─────────────────────┘
        │
        ▼
  Intrinsic code + scalar fallback + verified speedup
```

Two Gemini calls per request: one for intent extraction into a structured JSON schema, one for code synthesis. The second call injects relevant intrinsic semantics as context to reduce hallucination on non-trivial operations.

---

## Example Results

| Prompt | Generated Intrinsics | Speedup |
|--------|----------------------|---------|
| Element-wise multiply two float arrays | `_mm256_mul_ps` | 2.2x |
| FMA: a[i] = a[i] * b[i] + c[i] | `_mm256_fmadd_ps` | 3.4x |
| Sum float array skipping NaN values | `_mm256_cmp_ps` + `_mm256_blendv_ps` + `_mm256_add_ps` | 5.7x |

Scalar baseline compiled with `-fno-tree-vectorize` to prevent auto-vectorization from inflating speedup numbers. Benchmarks run over 100 iterations on 64K float arrays (fits in L3 cache) on AVX2-capable hardware.

---

## Stack

- **Gemini API** — intent parsing and code synthesis
- **FastAPI** — REST API orchestration
- **clang/g++** — sandboxed compilation and execution
- **Docker** — containerized toolchain
- **Google Cloud Run** — deployment

---

## Setup

### Prerequisites
- Python 3.11+
- g++ with AVX2 support (`grep avx2 /proc/cpuinfo`)
- Google Gemini API key

### Local

```bash
git clone https://github.com/yourusername/simd-agent.git
cd simd-agent
pip install -r requirements.txt
cp .env.example .env
# add your GENAI_API_KEY to .env
python app.py
```

Server runs at `http://localhost:8000`.

### Docker

```bash
docker build -t simd-agent .
docker run -p 8080:8080 -e GENAI_API_KEY=your_key_here simd-agent
```

---

## API

```bash
POST /optimize
Content-Type: application/json

{
  "prompt": "compute dot product of two float arrays using FMA"
}
```

Response:

```json
{
  "scalar_code": "...",
  "optimized_code": "...",
  "speedup": 3.4,
  "scalar_time_ms": 4.47,
  "simd_time_ms": 1.31,
  "compilation_output": "",
  "message": null
}
```

---

## Deployment

```bash
docker build -t us-central1-docker.pkg.dev/PROJECT_ID/simd-agent-repo/simd-agent:latest .
docker push us-central1-docker.pkg.dev/PROJECT_ID/simd-agent-repo/simd-agent:latest

gcloud run deploy simd-agent \
  --image=us-central1-docker.pkg.dev/PROJECT_ID/simd-agent-repo/simd-agent:latest \
  --platform=managed \
  --region=us-central1 \
  --allow-unauthenticated \
  --set-env-vars=GENAI_API_KEY=your_key_here \
  --memory=1Gi \
  --cpu=2 \
  --timeout=120
```

---

## Limitations

- Assumes `n` is a multiple of 8 (AVX2 processes 8 floats per register)
- Horizontal reduction operations (cross-lane) are harder for the model and may consume retries
- AVX-512 generation is possible but not yet the default target ISA

---

## License

MIT
