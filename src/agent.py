"""LangChain ReAct agent over fine-tuned Phi-3-mini, with a fallback
path through Llama-3.3-70B on the HF Inference API when no GPU is
available locally (i.e. the Mac demo path).

Two ways to construct the agent:

    build_agent(adapter_path="models/phi3-trustlens-lora")
        Loads Phi-3-mini-4k-instruct + the LoRA adapter in 4-bit.
        Requires bitsandbytes + a CUDA GPU. Use this on Colab and
        for the proper STEP 14 evaluation.

    build_agent()
        Uses meta-llama/Llama-3.3-70B-Instruct via HF Inference API.
        No GPU required. Use this for the local Streamlit demo.
"""
import os

from dotenv import load_dotenv
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate

from src.tools import ALL_TOOLS

load_dotenv()

REACT_PROMPT = PromptTemplate.from_template(
    """You are a fraud-detection analyst reviewing online job postings.

You have access to the following tools:

{tools}

Use exactly this format:

Question: the job posting to assess
Thought: think about which tool to call next
Action: one of [{tool_names}]
Action Input: the input to the action
Observation: the tool's result
... (this Thought/Action/Action Input/Observation block can repeat 2-3 times)
Thought: I now have enough evidence to write the final assessment.
Final Answer: a single JSON object with these exact keys:
  trust_score (integer 0-100, higher = safer),
  red_flags (list of short strings),
  risk_breakdown ({{"financial": int 0-100, "legitimacy": int 0-100, "data_privacy": int 0-100}}),
  action ("avoid" or "caution" or "safe"),
  reasoning (1-3 sentences).

Begin!

Question: {input}
Thought:{agent_scratchpad}"""
)


def _llm_phi3_with_adapter(adapter_path):
    """Load base Phi-3 in 4-bit + LoRA adapter as a LangChain LLM.
    Requires bitsandbytes — CUDA only.
    """
    import torch
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              BitsAndBytesConfig, pipeline)
    from peft import PeftModel
    from langchain_huggingface import HuggingFacePipeline

    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
    )
    base = AutoModelForCausalLM.from_pretrained(
        "microsoft/Phi-3-mini-4k-instruct",
        quantization_config=bnb, device_map="auto", trust_remote_code=True,
    )
    mdl = PeftModel.from_pretrained(base, adapter_path)
    tok = AutoTokenizer.from_pretrained(adapter_path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    pipe = pipeline(
        "text-generation", model=mdl, tokenizer=tok,
        max_new_tokens=512, do_sample=False, return_full_text=False,
    )
    return HuggingFacePipeline(pipeline=pipe)


def _llm_hf_api():
    """Llama-3.3-70B-Instruct via Groq's direct API (free, fast).

    We originally used HuggingFace Inference API but hit two walls:
    1. HF auto-routes Llama-3.3-70B to Groq, which only supports the
       'conversational' endpoint (not LangChain's default text-generation).
    2. The HF free tier rate-limits and 402s once monthly credits are gone.

    Groq's own free tier is much more generous (~30 RPM, 14400 RPD) and
    serves Llama-3.3-70B directly. Get a key at console.groq.com and
    set GROQ_API_KEY in .env.

    Falls back to ChatHuggingFace if no GROQ_API_KEY is available.
    """
    if os.environ.get("GROQ_API_KEY"):
        from langchain_groq import ChatGroq
        return ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0.2,
            max_tokens=512,
            api_key=os.environ["GROQ_API_KEY"],
        )
    from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("Set GROQ_API_KEY (preferred) or HF_TOKEN in .env")
    llm = HuggingFaceEndpoint(
        repo_id="meta-llama/Llama-3.3-70B-Instruct",
        task="conversational",
        max_new_tokens=512, temperature=0.2,
        huggingfacehub_api_token=token,
    )
    return ChatHuggingFace(llm=llm)


def build_agent(adapter_path: str = None) -> AgentExecutor:
    """Build a LangChain ReAct AgentExecutor.

    If adapter_path is given, loads base Phi-3 + LoRA adapter
    (the production path on Colab / CUDA). Otherwise uses
    Llama-3.3-70B via HF API (the Mac demo path).
    """
    llm = _llm_phi3_with_adapter(adapter_path) if adapter_path else _llm_hf_api()
    agent = create_react_agent(llm, ALL_TOOLS, REACT_PROMPT)
    return AgentExecutor(
        agent=agent,
        tools=ALL_TOOLS,
        verbose=False,
        max_iterations=4,
        handle_parsing_errors=True,
        return_intermediate_steps=False,
    )
