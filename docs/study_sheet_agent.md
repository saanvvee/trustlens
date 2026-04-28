# Study sheet — `src/agent.py`

This file builds the LangChain **ReAct agent** that runs at inference
time. The agent:
1. Receives a job posting.
2. Chooses tools to call (the 4 from `src/tools.py`).
3. Reads each tool's output as an `Observation`.
4. After 2–3 tool calls, writes a structured JSON assessment as the
   `Final Answer`.

ReAct = "Reasoning + Acting" (Yao et al., 2022). The model interleaves
*thoughts* (free-form reasoning) with *actions* (tool calls). It's
the same pattern OpenAI's "function calling" implements but with
plain text instead of structured tool-use APIs.

## Two backends

### `_llm_phi3_with_adapter(adapter_path)` — the production path
Loads `microsoft/Phi-3-mini-4k-instruct` in 4-bit (NF4) via
bitsandbytes, attaches the LoRA adapter from STEP 10 via PEFT,
wraps as a HuggingFace `pipeline("text-generation")`, then as a
LangChain `HuggingFacePipeline`. **Requires CUDA** — bitsandbytes
doesn't ship for macOS.

### `_llm_hf_api()` — the demo path
Uses `langchain_huggingface.HuggingFaceEndpoint` to call
`Llama-3.3-70B-Instruct` on the HF Inference API. No GPU required.
Used for the local Streamlit demo and as a fallback when the
adapter isn't available.

## `build_agent(adapter_path=None) -> AgentExecutor`
The single public entry point. If `adapter_path` is provided, uses
the Phi-3 path; otherwise uses the HF API path. Returns a
`LangChain AgentExecutor` configured with:

- `tools = ALL_TOOLS` from `src/tools.py`
- `max_iterations = 4` — caps the Thought / Action loop so a
  rambling model can't burn forever
- `handle_parsing_errors = True` — if the model emits malformed
  ReAct, LangChain feeds the error back as an Observation so the
  model can self-correct instead of crashing

## The `REACT_PROMPT`
Customised from the standard ReAct template. Two changes:

1. **Domain framing** — "You are a fraud-detection analyst" anchors
   the model's behaviour even when no tool has been called yet.
2. **Strict JSON Final Answer schema** — the standard ReAct prompt
   accepts free-form final answers; we enforce the same JSON schema
   the teacher emitted at STEP 8 so downstream code (pipeline.py)
   can `json.loads` the result without a separate "extract JSON"
   pass.

## Sample viva Q&A

**Q: Why a ReAct agent over a single-call prompt?**
A: A single-call prompt gives the model one shot to consider every
signal — it has to embed all the heuristic features and the kNN
fraud rate into the same prompt. The ReAct agent lets the model
*choose* which tools to consult based on what it sees in the
posting. If a posting has no email, the model skips
`check_email_domain`. If the salary is missing, it skips
`analyze_salary_realism`. This is closer to how a human analyst
works: read first, then look up specifics.

**Q: Why two backends — won't the demo behave differently from the
proper eval?**
A: Yes, deliberately. The Phi-3 + LoRA path is for **STEP 14
quantitative evaluation** (the rubric requires numbers from the
fine-tuned model). The Llama-API path is for the **demo** so the
viva grader can interact with a working system without a GPU. The
ReAct prompt and the tools are identical, so the *behaviour*
matches even though the underlying model differs. We document this
trade-off in the README.

**Q: Why `max_iterations=4`?**
A: Empirically, with our 4 tools, the model usually needs 2–3 calls
(features → company → salary or email). 4 gives a buffer for one
retry without letting an over-thinking model loop indefinitely. Hit
the cap → we still return whatever Final Answer the model has at
that point; the JSON parser at the pipeline layer handles partial
outputs gracefully.
