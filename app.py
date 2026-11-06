#!/usr/bin/env python
# coding: utf-8
"""
案例级 RAG 网页问答：FastAPI 后端 + 单页聊天界面

用法：
    python app.py build                 # 建索引（改了 docs/ 后重跑）
    uvicorn app:app --reload --port 8000

.env：
    DASHSCOPE_API_KEY   向量化 + 重排
    DEEPSEEK_API_KEY    生成
    DEEPSEEK_MODEL      可选，默认 deepseek-chat
"""

import os
import re
import sys
import json
import math
import base64
import hashlib
from contextlib import asynccontextmanager

import jieba
import pymupdf
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi
from dashscope import TextReRank, MultiModalConversation

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from langchain_core.documents import Document
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI

load_dotenv()

ROOT = os.path.dirname(os.path.abspath(__file__))
DOC_DIR = os.path.join(ROOT, "docs")
INDEX_DIR = os.path.join(ROOT, "case_index")
CHUNKS_FILE = os.path.join(INDEX_DIR, "chunks.jsonl")
HTML_FILE = os.path.join(ROOT, "index.html")

EMBED_MODEL = "text-embedding-v3"
EMBED_BATCH = 10          # text-embedding-v3 单次最多 10 条
RERANK_MODEL = "gte-rerank-v2"
LLM_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
CHUNK_MAX = 800
CHUNK_MIN = 120
RECALL_K = 20
FINAL_K = 5
MAX_EXPAND_CHARS = 6000   # 扩族熔断：父块超过该字数不整表展开，改拼最高分子块
TBL_MIN_ROWS = 2          # 表格误检兜底：少于 2 行放弃表格化，退回普通文本

MAX_RELATED_CHARS = 3000  # 图表关联佐证总长熔断
RELATED_TOP_K = 2         # 每个命中块最多关联的对侧图表块数
TBL_TITLE_RE = re.compile(r"【(?:表格片段\s*\d+/\d+\s*·\s*)?表：(.+?)】")
FIG_TITLE_RE = re.compile(r"【图·AI转述】([^\n]+)")

VL_MODEL = "qwen3-vl-plus"                      # 矢量图转述模型（qwen-vl-plus 已下线）
FIG_DESC_CACHE_FILE = os.path.join(ROOT, "figure_desc_cache.json")
FIG_MAX_DIST = 250        # 图题与矢量框中心距离上限（pt），配不上丢弃（宁缺勿错）
FIG_CAPTION_RE = re.compile(r"^(图|图表)\s*\d+\s*[：:.、]")
DOT_LEADER_RE = re.compile(r"\.{3,}|…+")   # 目录行（“图1：xxx……12”）不作图题
VL_PROMPT = """你是研报图表转述助手。请描述这张图中可见的内容：
1. 图中标注了数值的，照抄图中数值，不得换算或估算；
2. 未标注数值的趋势只能定性描述（如“先升后降”），不得编造具体数字；
3. 无法辨认的部分直接说明“无法辨认”，宁漏勿编；
4. 用简洁的中文输出，不超过150字。"""

# 哨兵表格块：@@TBL|table_uid=...|title=...@@ \n markdown \n @@TBL_END@@
TBL_BLOCK_RE = re.compile(r"^@@TBL\|(.+?)@@\n(.*?)\n@@TBL_END@@$", re.M | re.S)
# 哨兵图片块：@@FIG|uid@@ \n 【图·AI转述】图题 \n 描述 \n @@FIG_END@@
FIG_BLOCK_RE = re.compile(r"^@@FIG\|(.+?)@@\n(.*?)\n@@FIG_END@@$", re.M | re.S)

# 研报没有“第X条”，按章节标题切；条款类文档同样命中“第X条/第X章”
HEADING_RE = re.compile(
    r"(?m)^[ \t　]*(?:第[一二三四五六七八九十百零两\d]{1,6}[条章节]"
    r"|[一二三四五六七八九十]{1,3}、"
    r"|[（(][一二三四五六七八九十]{1,3}[)）])"
)
NOISE_RE = re.compile(
    r"(百度文库[^\n]*"
    r"|第\s*\d+\s*页\s*共\s*\d+\s*页"
    r"|请务必阅读正文之后的免责条款[^\n]*"
    r"|敬请阅读末页的重要说明[^\n]*)"
)
PAGE_NO_RE = re.compile(r"(?m)^[ \t　]*\d{1,3}[ \t　]*$")
PAGE_JUNK_RE = re.compile(r"(?m)^\s*\d{1,3}\s*/\s*\d{1,3}\s*\n?")  # "80 / 222" 页眉行


def normalize_md_table(md: str) -> str:
    """压缩 Markdown 表格，让 LLM 能按列可靠对齐。

    PDF 抽取的表格常在列间穿插全空的填充单元格（如 "| | 营业收入 | | | 57796 , |"），
    列号一多 LLM 就读错列。这里删除整列全空的列（表头与数据行同步删，对齐不变），
    并把纯数字单元格内的空格/尾逗号清掉（"57796 ," → "57796"）。
    """
    lines = md.split("\n")
    idx, rows = [], []
    for i, ln in enumerate(lines):
        if ln.startswith("|"):
            rows.append([c.strip() for c in ln.strip().strip("|").split("|")])
            idx.append(i)
    if len(rows) >= 2:
        width = max(len(r) for r in rows)
        # 分隔行（|---|---|）每列都非空，不参与全空列判断
        sep_flags = [all(re.fullmatch(r"-{3,}", c) for c in r if c) for r in rows]
        for r, is_sep in zip(rows, sep_flags):  # 单元格清理：纯数字去空格、去尾逗号
            if is_sep:
                continue
            for j, c in enumerate(r):
                if c and re.fullmatch(r"[\d.,%\s]+", c) and re.search(r"\d", c):
                    r[j] = re.sub(r"\s+", "", c).rstrip(",")
        data_rows = [r for r, s in zip(rows, sep_flags) if not s]
        keep = [c for c in range(width) if any(c < len(r) and r[c] for r in data_rows)]
        for i, r in zip(idx, rows):
            lines[i] = "| " + " | ".join(r[c] if c < len(r) else "" for c in keep) + " |"
    return "\n".join(lines)

PROMPT = """你是研报与制度文件问答助手。只能依据【片段】回答，不得使用片段之外的知识。

规则：
1. 每个结论后标注来源编号，如 [1]、[2]。
2. 金额、日期、比例、评级、目标价等数字必须原文照抄，不得换算或估算。
3. 片段不足以回答时，只回复：文档中未提及。不要猜测。
4. 预测、评级、观点要说明是哪家机构的判断，不要当成事实陈述。
5. 不要复述片段以外的内容。

【片段】
{context}

【问题】
{question}
"""


# ---------------- 解析与切块 ----------------

def clean_text(t: str) -> str:
    t = NOISE_RE.sub("", t)
    t = PAGE_NO_RE.sub("", t)
    t = PAGE_JUNK_RE.sub("", t)
    t = re.sub(r"[ \t\u3000]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def read_text_file(path: str) -> str:
    for enc in ("utf-8", "gbk"):
        try:
            with open(path, "r", encoding=enc) as f:
                return f.read()
        except UnicodeDecodeError:
            continue
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def table_to_markdown(tab) -> str | None:
    """表格转 Markdown。单元格内换行清洗、None 转空串、竖线转全角，防破坏行结构。"""

    def cell(c):
        if c is None:
            return ""
        return re.sub(r"\s*\n\s*", " ", str(c)).replace("|", "｜").strip()

    rows = [[cell(c) for c in r] for r in tab.extract()]
    rows = [r for r in rows if any(r)]
    if len(rows) < TBL_MIN_ROWS:
        return None
    ncol = max(len(r) for r in rows)
    if ncol < 2:  # 单列“表格”多为误检的长文本段落
        return None
    rows = [r + [""] * (ncol - len(r)) for r in rows]
    out = ["| " + " | ".join(rows[0]) + " |", "|" + "---|" * ncol]
    out += ["| " + " | ".join(r) + " |" for r in rows[1:]]
    return normalize_md_table("\n".join(out))


TITLE_NOISE_RE = re.compile(r"^(单位[:：]|币种|注[:：]|附注|√|□|\d+\s*$)")


def _guess_title(prev_lines, tbl_rect, pno) -> str:
    """表标题：表格上方（≤80pt）最近的非噪声短行（≤60字），找不到则降级“第X页表格”。"""
    for r, txt in reversed(prev_lines[-3:]):  # 最多回看 3 行，跳过“单位：…”等噪声行
        if tbl_rect.y0 - r.y1 > 80:
            break
        if len(txt) <= 60 and not TITLE_NOISE_RE.match(txt):
            return txt.replace("@@", "")
    return f"第{pno}页表格"


def load_fig_cache() -> dict:
    if os.path.exists(FIG_DESC_CACHE_FILE):
        with open(FIG_DESC_CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_fig_cache(cache: dict) -> None:
    with open(FIG_DESC_CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False)


def describe_figure_png(png_bytes: bytes) -> str:
    """PNG 发给 qwen-vl-plus 转述（prompt 已约束防幻觉）。"""
    b64 = base64.b64encode(png_bytes).decode("ascii")
    resp = MultiModalConversation.call(
        model=VL_MODEL,
        api_key=require_env("DASHSCOPE_API_KEY", "向量化 + 重排 + 图转述"),
        messages=[{"role": "user", "content": [
            {"image": f"data:image/png;base64,{b64}"},
            {"text": VL_PROMPT},
        ]}],
    )
    if resp.status_code != 200:
        raise RuntimeError(f"{resp.code}: {resp.message}")
    c = resp.output.choices[0].message.content
    if isinstance(c, list):
        text = "".join(p.get("text", "") for p in c if isinstance(p, dict))
    else:
        text = str(c)
    return text.strip()


def _center_dist(r: pymupdf.Rect, other: pymupdf.Rect) -> float:
    return math.hypot((r.x0 + r.x1) / 2 - (other.x0 + other.x1) / 2,
                      (r.y0 + r.y1) / 2 - (other.y0 + other.y1) / 2)


def collect_figures(page, pno, source, kept, table_rects, fig_cache, new_keys):
    """把图题与最近的矢量绘图框配对，抠图转述。

    返回 {kept 行下标: 图描述哨兵文本}，插到对应图题行之后。
    """
    caps = [(i, r, txt) for i, (r, txt) in enumerate(kept)
            if FIG_CAPTION_RE.match(txt) and not DOT_LEADER_RE.search(txt)]
    if not caps:
        return {}
    try:
        raw_rects = list(page.cluster_drawings())
    except Exception as e:
        print(f"[warn] {source} p{pno} cluster_drawings 失败：{e}")
        return {}

    # 过滤：过小、过扁、近整页装饰框、表格边框线
    page_area = max(page.rect.get_area(), 1)
    cands = []
    for r in raw_rects:
        area = r.get_area()
        if area < 100 or area > 0.6 * page_area:
            continue
        if r.height > 0 and (r.width / r.height > 20 or r.height / r.width > 20):
            continue
        if any(r.intersects(tr) for tr in table_rects):
            continue
        cands.append(r)

    caps.sort(key=lambda x: x[1].y0)  # 按 y 序配对，先到先得
    out = {}
    for i, cr, txt in caps:
        if not cands:
            break
        best = min(cands, key=lambda rr: _center_dist(rr, cr))
        if _center_dist(best, cr) > FIG_MAX_DIST:
            continue  # 配不上的框/图题丢弃，宁缺勿错
        cands.remove(best)
        key = hashlib.md5(f"{source}|{pno}|{txt}".encode("utf-8")).hexdigest()[:16]
        if key in fig_cache:
            desc = fig_cache[key]
        else:
            try:
                pix = page.get_pixmap(matrix=pymupdf.Matrix(2, 2), clip=best)
                desc = describe_figure_png(pix.tobytes("png"))
            except Exception as e:
                print(f"[warn] {source} p{pno} 图转述失败（{txt[:24]}…）：{e}")
                continue
            fig_cache[key] = desc
            new_keys.add(key)
        if "仅可见" in desc and "无其他" in desc:
            continue  # 误配到装饰线/空白框，宁缺勿错（描述已入缓存，不重复调用）
        out[i] = f"@@FIG|{key}@@\n【图·AI转述】{txt}\n{desc}\n@@FIG_END@@"
    return out


def pdf_page_text(page, pno: int, source: str, fig_cache=None, new_keys=None) -> str:
    """单页提取：有线表格转 Markdown（哨兵包裹），表格区域内的线性化文本剔除，
    图题配对矢量框抠图转述（哨兵包裹）。"""
    tables = []
    try:
        for i, tab in enumerate(page.find_tables().tables):
            md = table_to_markdown(tab)
            if md:
                tables.append((pymupdf.Rect(tab.bbox), md, i))
    except Exception as e:
        print(f"[warn] {source} p{pno} find_tables 失败：{e}")
    table_rects = [tr for tr, _, _ in tables]

    # 逐行重建文本，与表格 bbox 相交超半的行视为表格自身内容，跳过
    lines = []
    for b in page.get_text("dict")["blocks"]:
        if b.get("type") != 0:
            continue
        for ln in b["lines"]:
            txt = "".join(sp["text"] for sp in ln["spans"]).strip()
            if txt:
                lines.append((pymupdf.Rect(ln["bbox"]), txt))

    def overlaps(r: pymupdf.Rect, tr: pymupdf.Rect) -> bool:
        inter = r & tr
        return (not inter.is_empty) and inter.get_area() > 0.5 * max(r.get_area(), 1)

    kept = [(r, txt) for r, txt in lines if not any(overlaps(r, tr) for tr in table_rects)]
    kept.sort(key=lambda x: (round(x[0].y0, 1), x[0].x0))
    tables.sort(key=lambda x: x[0].y0)

    fig_at = {}
    if fig_cache is not None and new_keys is not None:
        fig_at = collect_figures(page, pno, source, kept, table_rects, fig_cache, new_keys)

    def sentinel(tbl_rect, md, idx, prev_line) -> str:
        tu = hashlib.md5(f"{source}|{pno}|{idx}".encode("utf-8")).hexdigest()[:12]
        title = _guess_title(prev_line, tbl_rect, pno).replace("\n", " ")
        return f"@@TBL|table_uid={tu}|title={title}@@\n{md}\n@@TBL_END@@"

    parts, prev_lines, ti = [], [], 0
    for ki, (r, txt) in enumerate(kept):
        while ti < len(tables) and tables[ti][0].y0 <= r.y0:
            tr, md, idx = tables[ti]
            parts.append(sentinel(tr, md, idx, prev_lines))
            ti += 1
        parts.append(txt)
        if ki in fig_at:  # 图描述紧跟图题行
            parts.append(fig_at[ki])
        prev_lines.append((r, txt))
    while ti < len(tables):  # 页尾表格
        tr, md, idx = tables[ti]
        parts.append(sentinel(tr, md, idx, prev_lines))
        ti += 1
    return clean_text("\n".join(parts))


def parse_file(path: str):
    """返回 (全文, 字符级页码表)。"""
    ext = os.path.splitext(path)[1].lower()
    text, page_map = "", []

    if ext == ".pdf":
        doc = pymupdf.open(path)
        source = os.path.basename(path)
        fig_cache, new_keys = load_fig_cache(), set()
        for pno, page in enumerate(doc, start=1):
            t = pdf_page_text(page, pno, source, fig_cache, new_keys)
            if not t:
                print(f"[warn] {source} p{pno} 无可提取文本（可能是扫描页）")
                continue
            text += t + "\n"
            page_map.extend([pno] * (len(t) + 1))
        if new_keys:
            save_fig_cache(fig_cache)
            print(f"{source}: 新转述 {len(new_keys)} 张矢量图 → figure_desc_cache.json")
    else:
        text = clean_text(read_text_file(path))
        page_map = [1] * len(text)

    return text, page_map


def split_long(seg: str, max_len: int):
    if len(seg) <= max_len:
        return [seg]
    parts, buf = [], ""
    for s in re.split(r"(?<=[。；！？\n])", seg):
        if buf and len(buf) + len(s) > max_len:
            parts.append(buf)
            buf = ""
        buf += s
    if buf.strip():
        parts.append(buf)
    return [p for p in parts if p.strip()]


def build_chunks(path: str):
    """返回 (docs, parents)。parents: table_uid -> 大表父块整表内容（仅映射，不入索引）。"""
    text, page_map = parse_file(path)
    source = os.path.basename(path)
    parents = {}

    docs, seen = [], set()

    def add_doc(content: str, piece_start: int, extra: dict | None = None) -> None:
        uid = hashlib.md5(re.sub(r"\s+", "", content).encode("utf-8")).hexdigest()
        if uid in seen:
            return
        seen.add(uid)
        page = page_map[min(piece_start, len(page_map) - 1)] if page_map else 1
        meta = {"source": source, "page": page, "uid": uid}
        if extra:
            meta.update(extra)
        docs.append(Document(page_content=content, metadata=meta))

    def emit(buf: str, buf_start: int) -> None:
        if not buf.strip():
            return
        offset = 0
        out = []  # (piece_start, content)
        for piece in split_long(buf, CHUNK_MAX):
            piece_start = buf_start + offset
            offset += len(piece)
            content = piece.strip()
            if not content:
                continue
            # 兜底切出的小尾巴并入前一块，避免内容丢失
            if out and len(content) < CHUNK_MIN:
                prev_start, prev = out[-1]
                out[-1] = (prev_start, prev + content)
            else:
                out.append((piece_start, content))
        for piece_start, content in out:
            add_doc(content, piece_start)

    def feed_normal(seg: str, seg_start: int, buf_state: dict) -> None:
        """普通文本段：沿用标题对齐 + CHUNK_MIN 合并的原有逻辑。"""
        if not seg.strip():
            return
        marks = sorted({0, len(seg), *[m.start() for m in HEADING_RE.finditer(seg)]})
        for i in range(len(marks) - 1):
            start = seg_start + marks[i]
            part = seg[marks[i]:marks[i + 1]]
            if not part.strip():
                continue
            if not buf_state["buf"]:
                buf_state["start"] = start
            buf_state["buf"] += part
            if len(buf_state["buf"].strip()) >= CHUNK_MIN:
                emit(buf_state["buf"], buf_state["start"])
                buf_state["buf"] = ""

    def flush(buf_state: dict) -> None:
        if buf_state["buf"]:
            emit(buf_state["buf"], buf_state["start"])
            buf_state["buf"] = ""

    def add_table(tu: str, title: str, md: str, start: int) -> None:
        head = f"【表：{title}】"
        full = f"{head}\n{md}"
        if len(full) <= CHUNK_MAX:
            # 小表：唯一块直接入库，扩族时用自身
            add_doc(full, start, {"type": "table", "table_uid": tu,
                                  "parent": True, "part": "1/1"})
            return
        # 大表：父块整表仅存映射不入索引；子块按行切分（超长单行不拆），表头重复 + 认亲标注
        parents[tu] = full
        lines = md.split("\n")
        header, rows = "\n".join(lines[:2]), lines[2:]

        def ann(i: int, n: int) -> str:
            tail = " · 续前块" if i > 1 else ""
            return f"【表格片段 {i}/{n} · 表：{title}{tail}】"

        budget = CHUNK_MAX - len(header) - max(len(ann(1, 1)), len(ann(2, 2))) - 2
        groups, buf, blen = [], [], 0
        for row in rows:
            if buf and blen + len(row) + 1 > budget:
                groups.append(buf)
                buf, blen = [], 0
            buf.append(row)
            blen += len(row) + 1
        if buf:
            groups.append(buf)
        n = len(groups)
        for i, g in enumerate(groups, 1):
            content = f"{ann(i, n)}\n{header}\n" + "\n".join(g)
            add_doc(content, start, {"type": "table", "table_uid": tu,
                                     "parent": False, "part": f"{i}/{n}"})

    def add_fig(fig_uid: str, content: str, start: int) -> None:
        # 图描述自带图题上下文，单独成块，不做父子
        add_doc(content, start, {"type": "figure", "ai_generated": True,
                                 "fig_uid": fig_uid})

    buf_state = {"buf": "", "start": 0}
    events = sorted(
        [(m.start(), m.end(), "table", m) for m in TBL_BLOCK_RE.finditer(text)]
        + [(m.start(), m.end(), "fig", m) for m in FIG_BLOCK_RE.finditer(text)],
        key=lambda e: e[0],
    )
    pos = 0
    for start, end, kind, m in events:
        feed_normal(text[pos:start], pos, buf_state)
        flush(buf_state)  # 哨兵块是语义边界，前面的普通文本先落块

        if kind == "table":
            # 哨兵解析：table_uid 不含 |，title 可能含 |，故只在第一个 | 处切一次
            payload = m.group(1)
            uid_part, _, rest = payload.partition("|")
            tu = uid_part[len("table_uid="):]
            title = rest[len("title="):] if rest.startswith("title=") else rest
            add_table(tu, title, m.group(2).strip(), start)
        else:
            add_fig(m.group(1), m.group(2).strip(), start)
        pos = end
    feed_normal(text[pos:], pos, buf_state)
    # 文档末尾不足 CHUNK_MIN 的残余段（如“六、联系方式”）也要写入
    flush(buf_state)

    return docs, parents


# ---------------- 索引 ----------------

def require_env(name: str, hint: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise SystemExit(f"缺少 {name}：请在 {os.path.join(ROOT, '.env')} 中填写（{hint}）")
    return value


def get_embeddings():
    return DashScopeEmbeddings(
        model=EMBED_MODEL,
        dashscope_api_key=require_env("DASHSCOPE_API_KEY", "向量化 + 重排"),
    )


def build_index():
    if not os.path.isdir(DOC_DIR):
        raise SystemExit(f"找不到语料目录：{DOC_DIR}")

    files = [
        os.path.join(DOC_DIR, f)
        for f in sorted(os.listdir(DOC_DIR))
        if os.path.splitext(f)[1].lower() in {".pdf", ".txt", ".md"}
    ]
    if not files:
        raise SystemExit("docs/ 下没有可解析的文件")

    docs, table_parents = [], {}
    for path in files:
        part, parents = build_chunks(path)
        print(f"{os.path.basename(path)}: {len(part)} chunks")
        docs.extend(part)
        table_parents.update(parents)

    print(f"合计 {len(docs)} 个片段（其中大表父块 {len(table_parents)} 张，仅存映射），开始向量化…")
    emb = get_embeddings()
    texts = [d.page_content for d in docs]
    vectors = []
    for i in range(0, len(texts), EMBED_BATCH):
        vectors.extend(emb.embed_documents(texts[i:i + EMBED_BATCH]))
        print(f"  已向量化 {min(i + EMBED_BATCH, len(texts))}/{len(texts)}")

    vs = FAISS.from_embeddings(
        list(zip(texts, vectors)),
        emb,
        metadatas=[d.metadata for d in docs],
    )

    os.makedirs(INDEX_DIR, exist_ok=True)
    vs.save_local(INDEX_DIR)
    with open(CHUNKS_FILE, "w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps({"text": d.page_content, **d.metadata},
                               ensure_ascii=False) + "\n")
    with open(os.path.join(INDEX_DIR, "table_parents.json"), "w", encoding="utf-8") as f:
        json.dump(table_parents, f, ensure_ascii=False)

    print(f"索引完成 → {INDEX_DIR}")


DISCLAIMER_RE = re.compile(
    r"^(?:免责声明|重要声明|分析师声明|特别声明|版权声明|请务必阅读)"
)


def is_disclaimer_block(d: dict) -> bool:
    """研报免责/声明块（每页重复、对问答零价值），整块过滤。"""
    if d.get("type") in ("table", "figure"):
        return False
    head = d["text"][:60]
    return bool(DISCLAIMER_RE.match(head) or "免责声明和披露" in head
                or "请务必阅读" in head and len(d["text"]) < 400)


def load_index():
    if not os.path.exists(CHUNKS_FILE):
        raise RuntimeError("索引不存在，请先执行：python app.py build")

    emb = get_embeddings()
    vs = FAISS.load_local(INDEX_DIR, emb, allow_dangerous_deserialization=True)

    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        docs = [json.loads(line) for line in f if line.strip()]

    parents_file = os.path.join(INDEX_DIR, "table_parents.json")
    table_parents = {}
    if os.path.exists(parents_file):
        with open(parents_file, "r", encoding="utf-8") as f:
            table_parents = json.load(f)

    # 后处理：清理页眉垃圾行 + 压缩 Markdown 表格（旧索引无需重建即可生效；
    # 向量基于原始文本，压缩只影响 BM25/LLM 阅读，语义 token 不变）
    for d in docs:
        d["text"] = PAGE_JUNK_RE.sub("", d["text"]).strip()
        if d.get("type") == "table":
            d["text"] = normalize_md_table(d["text"])
    table_parents = {tu: normalize_md_table(t) for tu, t in table_parents.items()}

    # 清理后可能出现大量"每页一条"的完全相同标题块，按 (source, text) 去重，
    # 防止同一垃圾文本以多个 uid 重复挤占召回槽位
    seen_st, deduped = set(), []
    for d in docs:
        k = (d["source"], d["text"])
        if k in seen_st or is_disclaimer_block(d):
            continue
        seen_st.add(k)
        deduped.append(d)
    docs = deduped

    bm25 = BM25Okapi([list(jieba.cut(d["text"])) for d in docs])
    return vs, docs, bm25, table_parents


# ---------------- 检索与生成 ----------------

def retrieve(query: str, vs, docs, bm25, table_parents=None,
             recall_k=RECALL_K, final_k=FINAL_K):
    """返回 (ranked hits, families)。
    families: table_uid -> [(rrf分数, doc)] 按分数降序，供扩族熔断挑选最高分子块。
    """
    by_uid = {d["uid"]: d for d in docs}

    vec_hits = [by_uid.get(d.metadata["uid"])
                for d in vs.similarity_search(query, k=recall_k)]
    bm_hits = bm25.get_top_n(list(jieba.cut(query)), docs, n=recall_k)

    score = {}
    for hits in (vec_hits, bm_hits):
        for rank, d in enumerate(hits):
            if not d:
                continue
            score[d["uid"]] = score.get(d["uid"], 0) + 1 / (60 + rank + 1)

    by_sc = sorted(score.items(), key=lambda kv: kv[1], reverse=True)

    # 聚簇（重排前）：同一 table_uid 家族只让分数最高的一个占名额，其余让位
    families = {}
    for uid, sc in by_sc:
        tu = by_uid[uid].get("table_uid")
        if tu:
            families.setdefault(tu, []).append((sc, by_uid[uid]))
    drop = {d["uid"] for members in families.values() for _, d in members[1:]}
    cands = [by_uid[u] for u, _ in by_sc if u not in drop][:recall_k]
    if not cands:
        return [], {}

    try:
        resp = TextReRank.call(
            model=RERANK_MODEL,
            query=query,
            documents=[c["text"] for c in cands],
            top_n=min(final_k, len(cands)),
            return_documents=False,
        )
        ranked = [(cands[r["index"]], float(r["relevance_score"]))
                  for r in resp.output.results]
    except Exception as e:
        print(f"[warn] 重排失败，退化为融合排序：{e}")
        ranked = [(c, 0.0) for c in cands[:final_k]]

    return ranked, families


def get_llm():
    return ChatOpenAI(
        model=LLM_MODEL,
        api_key=require_env("DEEPSEEK_API_KEY", "生成"),
        base_url="https://api.deepseek.com",
        temperature=0.1,
    )


def expand_table_hit(d: dict, families: dict, table_parents: dict) -> dict:
    """表格块扩族：命中子块 → 父块整表；父块超长则熔断拼最高分前 3 个子块。"""
    tu = d.get("table_uid")
    if not tu or d.get("parent") or tu not in table_parents:
        return d  # 非表格块 / 小表唯一块 / 无映射：原样
    parent = table_parents[tu]
    if len(parent) <= MAX_EXPAND_CHARS:
        return {**d, "text": parent}
    top = [m for _, m in families.get(tu, [])[:3]]
    return {**d, "text": "\n\n".join(m["text"] for m in top) if top else parent}


def _block_title(d: dict) -> str:
    m = TBL_TITLE_RE.match(d["text"]) or FIG_TITLE_RE.match(d["text"])
    title = m.group(1) if m else ""
    return title.replace(" · 续前块", "").strip()


def _word_overlap(a: str, b: str) -> int:
    ta = {w.strip() for w in jieba.cut(a) if len(w.strip()) >= 2}
    tb = {w.strip() for w in jieba.cut(b) if len(w.strip()) >= 2}
    return len(ta & tb)


def find_related_blocks(d: dict, query: str, page_index: dict, used_uids: set) -> list:
    """为命中块找同源图表关联块（双向：图↔表）。

    范围：同 source 的 [page-1, page, page+1]；排序：同页优先 + 与
    (query + 原块标题) 的词面重合度；邻页块毫无词面交集则不关联。
    大表：同 table_uid 子块归并，取与 query 词面重合最高的 1 个子块。
    """
    typ = d.get("type")
    if typ not in ("table", "figure"):
        return []
    other = "figure" if typ == "table" else "table"
    probe = query + " " + _block_title(d)

    cands = []
    for p in (d["page"] - 1, d["page"], d["page"] + 1):
        for c in page_index.get((d["source"], p), []):
            if c.get("type") != other or c["uid"] in used_uids:
                continue
            if c.get("table_uid") and c.get("table_uid") == d.get("table_uid"):
                continue  # 同一张表的其他片段不算对侧佐证
            cands.append(c)
    if not cands:
        return []

    if other == "table":  # 大表多个子块只保留最优的一个候选
        best = {}  # table_uid 或独立块 uid -> (词面重合, doc)
        for c in cands:
            ov = _word_overlap(probe, c["text"])
            tu = c.get("table_uid")
            key = tu if (tu and not c.get("parent")) else f"solo:{c['uid']}"
            cur = best.get(key)
            prefer_first = (cur is not None and ov == cur[0]
                            and c.get("part", "").startswith("1/")
                            and not cur[1].get("part", "").startswith("1/"))
            if cur is None or ov > cur[0] or prefer_first:
                best[key] = (ov, c)
        cands = [(ov, c) for ov, c in best.values()]
    else:
        cands = [(_word_overlap(probe, c["text"]), c) for c in cands]

    ranked = []
    for ov, c in cands:
        same_page = c["page"] == d["page"]
        if not same_page and ov == 0:
            continue  # 邻页且毫无词面交集：不引入跨页噪音
        ranked.append(((100 if same_page else 0) + ov, c["uid"], c))
    ranked.sort(key=lambda x: (-x[0], x[1]))
    return [c for _, _, c in ranked[:RELATED_TOP_K]]


def answer_question(query: str, hits, families=None, table_parents=None, docs=None):
    families = families or {}
    table_parents = table_parents or {}
    expanded = [expand_table_hit(d, families, table_parents) for d, _ in hits]

    # 图表关联佐证（双向图↔表）：FINAL_K 之外附加，不挤占主名额
    related_map = {}  # 主命中块 uid -> [关联块]
    if docs:
        page_index = {}
        for d in docs:
            page_index.setdefault((d["source"], d["page"]), []).append(d)
        used = {d["uid"] for d, _ in hits}  # 主命中块去重
        budget = MAX_RELATED_CHARS
        for d, _ in hits:
            rels = []
            for c in find_related_blocks(d, query, page_index, used):
                if len(c["text"]) > budget:
                    continue  # 全局总长熔断
                budget -= len(c["text"])
                used.add(c["uid"])  # 已被关联过的对侧块不重复附加
                rels.append(c)
            if rels:
                related_map[d["uid"]] = rels
    related_count = sum(len(v) for v in related_map.values())

    # 佐证紧贴触发它的主命中块正下方，子编号如 [1a] 方便 LLM 就近对照引用
    sub_ids = "abcdef"
    lines = []
    for i, d in enumerate(expanded, 1):
        lines.append(f"[{i}] (来源：{d['source']} 第{d['page']}页)\n{d['text']}")
        for j, c in enumerate(related_map.get(d["uid"], [])):
            lines.append(f"[{i}{sub_ids[j]}] [关联图表佐证] "
                         f"(来源：{c['source']} 第{c['page']}页)\n{c['text']}")
    context = "\n\n".join(lines)
    text = get_llm().invoke(PROMPT.format(context=context, question=query)).content

    # 依据卡片仍显示命中子块自己的页码/相关度
    sources = [
        {
            "id": i,
            "source": d["source"],
            "page": d["page"],
            "score": round(s, 4),
            "text": d["text"][:300],
        }
        for i, (d, s) in enumerate(hits, 1)
    ]
    return text, sources, related_count


# ---------------- FastAPI ----------------

class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=500)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.rag = None
    try:
        app.state.rag = load_index()
        print(f"索引已加载：{len(app.state.rag[1])} 个片段")
    except Exception as e:
        print(f"[warn] 索引加载失败：{e}")
    yield
    app.state.rag = None


app = FastAPI(title="研报问答助手", lifespan=lifespan)


@app.get("/")
def home():
    if not os.path.exists(HTML_FILE):
        raise HTTPException(status_code=500, detail="缺少 index.html")
    return FileResponse(HTML_FILE)


@app.get("/api/health")
def health():
    rag = getattr(app.state, "rag", None)
    return {
        "status": "ok" if rag else "no_index",
        "chunks": len(rag[1]) if rag else 0,
        "index_ready": rag is not None,
    }


@app.post("/api/ask")
def ask(req: AskRequest):
    rag = getattr(app.state, "rag", None)
    if rag is None:
        raise HTTPException(status_code=503, detail="索引未就绪，请先执行 python app.py build")

    query = req.question.strip()
    if not query:
        raise HTTPException(status_code=400, detail="问题不能为空")

    hits, families = retrieve(query, *rag)
    if not hits:
        return {"answer": "文档中未提及", "sources": [], "related_count": 0}

    text, sources, related_count = answer_question(query, hits, families, rag[3], rag[1])
    return {"answer": text, "sources": sources, "related_count": related_count}


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "build":
        build_index()
    else:
        print(__doc__)
