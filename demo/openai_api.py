"""OpenAI 兼容 API 服务（FastAPI 包装）。

支持 /v1/chat/completions 子集，可选 RAG 注入与护栏。
"""

from __future__ import annotations

import argparse
import asyncio
import time
import uuid

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

from legal_lm.data.prompting import DEFAULT_SYSTEM_PROMPT
from legal_lm.safety.guardrails import Guardrail


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    temperature: float = 0.1
    max_tokens: int = 768
    stream: bool = False


def _build_engine(model_path: str, use_vllm: bool):
    if use_vllm:
        try:
            from vllm import LLM, SamplingParams
        except ImportError as exc:  # pragma: no cover
            raise ImportError("Install vLLM: pip install vllm") from exc

        llm = LLM(model=model_path, dtype="bfloat16")

        def _generate(prompt: str, max_tokens: int, temperature: float) -> str:
            sp = SamplingParams(temperature=temperature, max_tokens=max_tokens)
            return llm.generate([prompt], sp)[0].outputs[0].text

        return _generate

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, trust_remote_code=True
    ).eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    def _generate(prompt: str, max_tokens: int, temperature: float) -> str:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=temperature > 0,
                temperature=max(temperature, 1e-5),
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    return _generate


def _format_messages(messages: list[ChatMessage], rag_block: str = "") -> str:
    parts = []
    for msg in messages:
        body = msg.content
        if msg.role == "user" and rag_block:
            body = body + "\n\n" + rag_block
            rag_block = ""
        parts.append(f"<|{msg.role}|>\n{body}")
    parts.append("<|assistant|>\n")
    return "\n".join(parts)


def build_app(args: argparse.Namespace) -> FastAPI:
    generate = _build_engine(args.model_path, use_vllm=args.use_vllm)
    retriever = None
    if args.inject_rag:
        from legal_lm.data.rag import StatuteRetriever

        retriever = StatuteRetriever(args.statute_index_dir)
    guard = Guardrail() if args.enable_guardrails else None

    app = FastAPI(title="legal-lm-cn")

    @app.get("/v1/models")
    async def list_models():
        return {"object": "list", "data": [{"id": args.model_path, "object": "model"}]}

    @app.post("/v1/chat/completions")
    async def chat_completions(req: ChatCompletionRequest):
        if not any(m.role == "system" for m in req.messages):
            req.messages.insert(0, ChatMessage(role="system", content=DEFAULT_SYSTEM_PROMPT))

        if guard is not None:
            for msg in req.messages:
                if msg.role == "user":
                    msg.content = guard.pre_input(msg.content).text

        rag_block = ""
        if retriever is not None:
            last_user = next((m for m in reversed(req.messages) if m.role == "user"), None)
            if last_user is not None:
                rag_block = retriever.render_block(last_user.content, top_k=5)

        prompt = _format_messages(req.messages, rag_block=rag_block)
        loop = asyncio.get_event_loop()
        text = await loop.run_in_executor(
            None, generate, prompt, req.max_tokens, req.temperature
        )
        if guard is not None:
            text = guard.post_output(text).text

        return {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": req.model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "stop",
                }
            ],
        }

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="OpenAI 兼容法律 LM 服务")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--use-vllm", action="store_true")
    parser.add_argument("--inject-rag", action="store_true")
    parser.add_argument("--statute-index-dir", default="data/statute_index")
    parser.add_argument("--enable-guardrails", action="store_true")
    args = parser.parse_args()
    uvicorn.run(build_app(args), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
