#!/usr/bin/env python
# coding: utf-8
"""
案例级 RAG 控制台问答（复用 app.py 的索引与检索逻辑）

用法：
    python cli.py            # 交互问答（索引不存在时提示先建）
    python cli.py build      # 建索引
    python cli.py "问题"     # 单次问答后退出

.env 与 app.py 共用：
    DASHSCOPE_API_KEY   向量化 + 重排
    DEEPSEEK_API_KEY    生成
"""

import sys

# Windows 控制台中文输出
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from app import build_index, load_index, retrieve, answer_question, FINAL_K


def ask(query: str, rag) -> None:
    hits = retrieve(query, *rag)
    if not hits:
        print("\n文档中未提及\n")
        return

    text, sources = answer_question(query, hits)

    print("\n" + "=" * 60)
    print(text)
    print("-" * 60)
    for s in sources:
        print(f"[{s['id']}] {s['source']} 第{s['page']}页 "
              f"(相关度 {s['score']:.4f})")
    print("=" * 60 + "\n")


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "build":
        build_index()
        return

    # 单次问答模式：python cli.py "问题"
    if len(sys.argv) > 1:
        rag = load_index()
        ask(sys.argv[1], rag)
        return

    # 交互模式
    print("索引加载中…")
    rag = load_index()
    print(f"就绪：{len(rag[1])} 个片段。输入问题开始提问，输入 q 或 exit 退出。\n")

    while True:
        try:
            query = input("问题> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见")
            break

        if not query:
            continue
        if query.lower() in {"q", "exit", "quit"}:
            print("再见")
            break

        ask(query, rag)


if __name__ == "__main__":
    main()
