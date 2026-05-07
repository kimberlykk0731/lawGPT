"""Gradio Web Demo：调用 OpenAI 兼容 API 服务，左侧输入案件、右侧法条侧栏。"""

from __future__ import annotations

import argparse
import json

import gradio as gr
import httpx


def _make_client(base_url: str, model: str, timeout: float = 120.0):
    client = httpx.Client(base_url=base_url, timeout=timeout)

    def _retrieve(query: str, top_k: int) -> str:
        return ""  # API 服务侧已注入；此处仅作为占位

    def _chat(history: list[tuple[str, str]], user_text: str) -> tuple[list[tuple[str, str]], str]:
        messages = []
        for u, a in history:
            messages.append({"role": "user", "content": u})
            messages.append({"role": "assistant", "content": a})
        messages.append({"role": "user", "content": user_text})
        payload = {
            "model": model,
            "messages": messages,
            "temperature": 0.1,
            "max_tokens": 768,
        }
        resp = client.post("/chat/completions", content=json.dumps(payload).encode("utf-8"),
                           headers={"Content-Type": "application/json"})
        resp.raise_for_status()
        data = resp.json()
        reply = data["choices"][0]["message"]["content"]
        history = history + [(user_text, reply)]
        return history, ""

    return _chat, _retrieve


def main() -> None:
    parser = argparse.ArgumentParser(description="Gradio Web Demo")
    parser.add_argument("--api-base", default="http://127.0.0.1:8001/v1")
    parser.add_argument("--model", default="legal-lm-cn")
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    chat, _retrieve = _make_client(args.api_base, args.model)

    with gr.Blocks(title="Legal-LM-CN") as demo:
        gr.Markdown("# 中文法律分析助手\n输入案件事实，模型输出结构化法律分析（结论/法律依据/推理摘要/风险提示）。")
        chatbot = gr.Chatbot(height=520, label="对话")
        with gr.Row():
            user_box = gr.Textbox(label="案件事实 / 法律问题", lines=4, scale=4)
            send_btn = gr.Button("发送", variant="primary", scale=1)
        clear_btn = gr.Button("清空对话")
        send_btn.click(chat, inputs=[chatbot, user_box], outputs=[chatbot, user_box])
        user_box.submit(chat, inputs=[chatbot, user_box], outputs=[chatbot, user_box])
        clear_btn.click(lambda: ([], ""), outputs=[chatbot, user_box])

    demo.launch(share=args.share)


if __name__ == "__main__":
    main()
