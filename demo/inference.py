"""CLI 推理：交互式法律问答，可选 RAG 注入与护栏。"""

from __future__ import annotations

import argparse

from legal_lm.data.prompting import DEFAULT_SYSTEM_PROMPT
from legal_lm.safety.guardrails import Guardrail


def _build_engine(model_path: str, use_vllm: bool):
    if use_vllm:
        try:
            from vllm import LLM, SamplingParams
        except ImportError as exc:  # pragma: no cover
            raise ImportError("Install vLLM: pip install vllm") from exc

        llm = LLM(model=model_path, dtype="bfloat16")

        def _generate(prompt: str, max_new_tokens: int, temperature: float) -> str:
            sp = SamplingParams(temperature=temperature, max_tokens=max_new_tokens)
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

    def _generate(prompt: str, max_new_tokens: int, temperature: float) -> str:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0,
                temperature=max(temperature, 1e-5),
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        return tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    return _generate


def _maybe_retriever(inject_rag: bool, statute_index_dir: str, top_k: int):
    if not inject_rag:
        return None
    from legal_lm.data.rag import StatuteRetriever

    retriever = StatuteRetriever(statute_index_dir)
    return lambda q: retriever.render_block(q, top_k=top_k)


def _format_prompt(system_prompt: str, user_text: str, rag_block: str = "") -> str:
    body = user_text + ("\n\n" + rag_block if rag_block else "")
    return f"<|system|>\n{system_prompt}\n<|user|>\n{body}\n<|assistant|>\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="法律 LM CLI 推理")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--use-vllm", action="store_true")
    parser.add_argument("--inject-rag", action="store_true")
    parser.add_argument("--statute-index-dir", default="data/statute_index")
    parser.add_argument("--rag-top-k", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT)
    args = parser.parse_args()

    generate = _build_engine(args.model_path, use_vllm=args.use_vllm)
    retrieve = _maybe_retriever(args.inject_rag, args.statute_index_dir, args.rag_top_k)
    guard = Guardrail()

    print("[inference] 输入 'exit' 退出。")
    while True:
        try:
            user = input("用户> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not user or user.lower() == "exit":
            break
        pre = guard.pre_input(user)
        rag_block = retrieve(pre.text) if retrieve else ""
        prompt = _format_prompt(args.system_prompt, pre.text, rag_block=rag_block)
        raw = generate(prompt, args.max_new_tokens, args.temperature)
        post = guard.post_output(raw)
        print(f"\n助手> {post.text}\n")


if __name__ == "__main__":
    main()
