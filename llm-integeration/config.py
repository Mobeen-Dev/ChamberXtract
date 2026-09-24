"""
config.py: single place to tune the Ollama run. Edit values here, not in the runner.

Prompt placeholders use {{name}} syntax (double braces), so literal { } in your
prompts (JSON examples, code) never collide with substitution.
Every placeholder in SYSTEM_PROMPT / USER_PROMPT_TEMPLATE must be supplied either
by DEFAULT_VARIABLES below or at call time (CLI --var / --file, or run(variables=...)).
"""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# ---------------------------------------------------------------- connection
OLLAMA_HOST = "http://localhost:11434"   # default Ollama server address
MODEL = "qwen3.5:4b"                     # must match a name shown by `ollama list`
REQUEST_TIMEOUT_S = 600                  # per-request socket timeout (small models on CPU can be slow)
MAX_RETRIES = 2                          # retries for connection errors / HTTP 5xx only
RETRY_BACKOFF_S = 2.0                    # sleep = backoff * 2**attempt between retries
KEEP_ALIVE = "10m"                       # how long Ollama keeps the model in memory after a call

# Thinking mode (Ollama "think" parameter):
#   None  -> parameter omitted, model/Ollama default behaviour (safest)
#   True  -> ask for reasoning trace (saved separately in the output JSON)
#   False -> disable reasoning (faster, fewer tokens)
# If the model rejects the parameter, Ollama returns an HTTP 400 and the runner reports it.
THINK = False

# ------------------------------------------------------ sampling / generation
# Passed verbatim as Ollama "options". Remove a key to fall back to the model's default.
OPTIONS = {
    "temperature": 0.3,      # lower = more deterministic
    "top_p": 0.9,
    "top_k": 40,
    "repeat_penalty": 1.1,
    "num_ctx": 8192,         # context window in tokens (more = more RAM)
    "num_predict": 2048,     # max generated tokens (-1 = unlimited). Thinking tokens count too.
    "seed": 42,              # fixed seed => reproducible output at a given temperature
}

# --------------------------------------------------------------------- output
OUTPUT_DIR = BASE_DIR / "responses"      # one JSON file per run is written here

# ------------------------------------------------------------- master prompts
SYSTEM_PROMPT = """\
You are {{role}}.
Tone: {{tone}}.
Output format: {{output_format}}.
Rules:
- Be accurate. If information is missing or ambiguous, say so instead of guessing.
- Do not add preambles, apologies, or closing remarks.
"""

USER_PROMPT_TEMPLATE = """\
Task:
{{task}}

Input:
<<<
{{input_text}}
>>>
"""

# Values used when a placeholder is not supplied at call time.
# Placeholders with NO default here (task, input_text) are mandatory per run.
DEFAULT_VARIABLES = {
    "role": "a precise technical assistant",
    "tone": "concise and direct",
    "output_format": "plain text",
}
