# Self-Consistency in Large Language Model Reasoning

# User Manual

## Install
If running using a external model provider:
```bash
git clone https://github.com/zyanai-pub/self-consistency-in-LLMs.git
cd self-consistency-in-LLMs
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```


```bash
pip install python-dotenv datasets "staticfg>=0.9.5" "litellm>=1.0.0" google-generativeai tenacity
```

## Configuration

Create `.env` in the repository root.

```env
# local | hybrid | remote
EXECUTION_MODE=local

# needed for hybrid and remote
GROQ_API_KEY=...
GOOGLE_API_KEY=...
```

## Running on Colab

The quickest path to a full local run on a free T4. Open
`src/tts_eval_colab.ipynb` in Colab (File > Open notebook > GitHub, or upload the file),
set the runtime to T4 GPU, and run the cells top to bottom. It clones this repository
itself, starts both vLLM servers, and writes results to `results/colab` inside the
cloned copy.

## Running locally

Start both servers first, each in its own shell. These are the models and ports
`run_evaluation.py` expects in `local` mode.

```bash
vllm serve Qwen/Qwen2.5-3B-Instruct --port 8000
```

```bash
vllm serve Qwen/Qwen2.5-0.5B-Instruct --port 8001
```

Then, from the repository root:

```bash
python -m src.run_evaluation
```

It has to be `-m`. `python src/run_evaluation.py` fails with
`ModuleNotFoundError: No module named 'src'`, because the module imports `src.*` before
it fixes `sys.path`.

For `hybrid`, only the `:8000` server is needed. For `remote`, neither is: set
`EXECUTION_MODE=remote`, put a Groq key in `.env`, and run the same command.

## Output

Written to `results/`, stamped with a `YYYYMMDD-HHMMSS` run id:

- `<run_id>_<model>_<strategy>.json`, one per model and strategy
- `<run_id>_summary.json`, accuracy and counts per strategy

A summary table is also printed at the end of the run.

## Running remotely

from the repository root:

```bash
python -m src.run_evaluation
```

As shipped, remote mode uses `groq/llama-3.3-70b-versatile` for System-2 and
`groq/llama-3.1-8b-instant` for System-1. Both are set in the `else` branch near the top
of `src/run_evaluation.py`. `GOOGLE_API_KEY` is read and forwarded to litellm but no
default model uses it, so you only need it if you swap a Gemini model into `MODELS`.

## Tests

```bash
pip install -r test_requirements.txt
pytest
```

