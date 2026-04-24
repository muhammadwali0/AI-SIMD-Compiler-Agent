import os
import subprocess
import tempfile
import json
import re
from typing import Dict, Any, Optional
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from google import genai
from dotenv import load_dotenv

load_dotenv()

GENAI_API_KEY = os.getenv("GENAI_API_KEY")
if not GENAI_API_KEY:
    print("Warning: GENAI_API_KEY not found in environment.")

client = genai.Client(api_key=GENAI_API_KEY)

app = FastAPI(title="AI SIMD Compiler Agent Orchestrator")

# Mount static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")


class OptimizationRequest(BaseModel):
    prompt: str


class OptimizationResponse(BaseModel):
    scalar_code: Optional[str] = None
    optimized_code: Optional[str] = None  # Renamed from simd_code for alignment
    speedup: Optional[float] = None
    scalar_time_ms: Optional[float] = None
    simd_time_ms: Optional[float] = None
    compilation_output: Optional[str] = ""
    message: Optional[str] = None  # Added for error handling


PROMPT_TEMPLATE = """
You are a Senior Systems Engineer specializing in AVX2 optimization.
User request: {user_prompt}

Generate two C++ functions for the requested operation:
1. `void scalar_version(float* a, float* b, float* c, int n)` - A standard scalar implementation.
    IMPORTANT: Add `#pragma GCC optimize("no-tree-vectorize")` on the line directly above the function definition.
2. `void simd_version(float* a, float* b, float* c, int n)` - An optimized version using AVX2 intrinsics (`immintrin.h`).

Assume `n` is a multiple of 8 for simplicity. The arrays are float pointers.

Return ONLY a JSON object with two keys: "scalar_func" and "simd_func". 
Ensure the code is valid C++ and does not include any other text or markdown formatting.
IMPORTANT: Always use `_mm256_loadu_ps` and `_mm256_storeu_ps` (unaligned variants).
Never use `_mm256_load_ps` or `_mm256_store_ps` as the benchmark harness does not guarantee 32-byte alignment.
"""

BENCHMARK_HARNESS_TEMPLATE = """
#include <iostream>
#include <chrono>
#include <immintrin.h>
#include <algorithm>
#include <cstdlib>
#include <cstring>

{scalar_func}

{simd_func}

int main() {{
    const int n = 1 << 16;
    float* a = (float*)aligned_alloc(32, n * sizeof(float));
    float* b = (float*)aligned_alloc(32, n * sizeof(float));
    float* c_scalar = (float*)aligned_alloc(32, n * sizeof(float));
    float* c_simd = (float*)aligned_alloc(32, n * sizeof(float));

    std::fill(a, a + n, 1.0f);
    std::fill(b, b + n, 2.0f);
    std::fill(c_scalar, c_scalar + n, 0.0f);
    std::fill(c_simd, c_simd + n, 0.0f);

    // Warm up
    scalar_version(a, b, c_scalar, n);
    simd_version(a, b, c_simd, n);

    // Benchmark Scalar
    auto start_scalar = std::chrono::high_resolution_clock::now();
    for(int i=0; i<100; ++i) {{
        scalar_version(a, b, c_scalar, n);
    }}
    auto end_scalar = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> diff_scalar = end_scalar - start_scalar;

    // Benchmark SIMD
    auto start_simd = std::chrono::high_resolution_clock::now();
    for(int i=0; i<100; ++i) {{
        simd_version(a, b, c_simd, n);
    }}
    auto end_simd = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> diff_simd = end_simd - start_simd;

    // Verification
    bool match = true;
    for(int i=0; i<n; ++i) {{
        if(std::abs(c_scalar[i] - c_simd[i]) > 1e-5) {{
            match = false;
            break;
        }}
    }}

    free(a); free(b); free(c_scalar); free(c_simd);

    if (!match) {{
        std::cerr << "Error: Results do not match!" << std::endl;
        return 1;
    }}

    std::cout << diff_scalar.count() << std::endl;
    std::cout << diff_simd.count() << std::endl;

    return 0;
}}
"""


def clean_json_response(text: str) -> Dict[str, str]:
    text = re.sub(r"```json\s?|\s?```", "", text).strip()
    return json.loads(text)


@app.get("/")
async def serve_index():
    return FileResponse("static/index.html")


@app.post("/optimize", response_model=OptimizationResponse)
async def optimize(request: OptimizationRequest):
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=PROMPT_TEMPLATE.format(user_prompt=request.prompt),
        )
        generated_data = clean_json_response(response.text)

        scalar_func = generated_data["scalar_func"]
        simd_func = generated_data["simd_func"]

        harness_code = BENCHMARK_HARNESS_TEMPLATE.format(
            scalar_func=scalar_func, simd_func=simd_func
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            cpp_file_path = os.path.join(tmpdir, "bench.cpp")
            bin_file_path = os.path.join(tmpdir, "bench")

            with open(cpp_file_path, "w") as f:
                f.write(harness_code)

            # Compilation step
            MAX_RETRIES = 3
            last_error = ""

            for attempt in range(MAX_RETRIES):
                # regenerate if this is a retry
                if attempt > 0:
                    retry_prompt = f"""
            The previous code had a compiler error. Fix it and return only the corrected JSON.

            Original request: {request.prompt}
            Compiler error:
            {last_error}

            Return ONLY a JSON object with keys "scalar_func" and "simd_func". No markdown.
            """
                    response = client.models.generate_content(
                        model="gemini-2.0-flash", contents=retry_prompt
                    )
                    generated_data = clean_json_response(response.text)
                    scalar_func = generated_data["scalar_func"]
                    simd_func = generated_data["simd_func"]
                    harness_code = BENCHMARK_HARNESS_TEMPLATE.format(
                        scalar_func=scalar_func, simd_func=simd_func
                    )
                    with open(cpp_file_path, "w") as f:
                        f.write(harness_code)

                compile_cmd = [
                    "g++",
                    "-O3",
                    "-mavx2",
                    "-mfma",
                    cpp_file_path,
                    "-o",
                    bin_file_path,
                ]
                compile_proc = subprocess.run(
                    compile_cmd, capture_output=True, text=True
                )

                if compile_proc.returncode == 0:
                    break

                last_error = compile_proc.stderr
                if attempt == MAX_RETRIES - 1:
                    return OptimizationResponse(
                        message=f"Compiler Error after {MAX_RETRIES} attempts:\n{last_error}"
                    )

            # Execution step
            try:
                run_proc = subprocess.run(
                    [bin_file_path], capture_output=True, text=True, check=True
                )
            except subprocess.CalledProcessError as e:
                return OptimizationResponse(message=f"Runtime Error: {e.stderr}")

            lines = run_proc.stdout.strip().splitlines()
            if len(lines) < 2:
                return OptimizationResponse(
                    message="Unexpected benchmark output format"
                )

            scalar_time = float(lines[0])
            simd_time = float(lines[1])
            speedup = scalar_time / simd_time if simd_time > 0 else 0.0

            return OptimizationResponse(
                scalar_code=scalar_func,
                optimized_code=simd_func,
                speedup=speedup,
                scalar_time_ms=scalar_time,
                simd_time_ms=simd_time,
                compilation_output=compile_proc.stdout,
            )

    except Exception as e:
        return OptimizationResponse(message=f"System Error: {str(e)}")


@app.get("/cpuinfo")
async def cpuinfo():
    result = subprocess.run(
        ["grep", "-m1", "flags", "/proc/cpuinfo"], capture_output=True, text=True
    )
    return {"flags": result.stdout}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
