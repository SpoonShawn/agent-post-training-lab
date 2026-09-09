import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KNOWLEDGE_DIR = ROOT / "data" / "knowledge"

TOKEN_PATTERN = re.compile(
    r"[a-zA-Z]+|\d+(?:\.\d+)+|[\u4e00-\u9fff]+"
)
STOP_WORDS = {
    "一个",
    "一下",
    "什么",
    "是否",
    "当前",
    "帮我",
    "怎么",
    "怎样",
    "版本",
    "问题",
    "请问",
    "这个",
}


def load_docs():
    docs = []

    for filename in [
        "build_notes.json",
        "ui_guide.json",
        "incident_history.json",
    ]:
        path = KNOWLEDGE_DIR / filename

        with open(path, "r", encoding="utf-8") as f:
            items = json.load(f)

        for item in items:
            docs.append({
                "source": filename,
                "content": item
            })

    return docs


def _search_terms(query: str):
    """Extract useful Latin/version tokens and short Chinese character grams."""

    terms = set()
    for token in TOKEN_PATTERN.findall(query.lower()):
        if token in STOP_WORDS:
            continue

        if re.fullmatch(r"[\u4e00-\u9fff]+", token):
            if len(token) <= 4:
                terms.add(token)
            else:
                for width in (2, 3, 4):
                    terms.update(
                        token[index:index + width]
                        for index in range(len(token) - width + 1)
                        if token[index:index + width] not in STOP_WORDS
                    )
        else:
            terms.add(token)

    return terms


def search_knowledge(query: str, top_k: int = 3):
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query 必须是非空字符串。")
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
        raise ValueError("top_k 必须是正整数。")

    query_words = _search_terms(query)
    scored = []

    for doc in load_docs():
        text = json.dumps(doc["content"], ensure_ascii=False).lower()

        matched_terms = [
            word
            for word in query_words
            if word in text
        ]
        score = sum(max(1, len(word) - 1) for word in matched_terms)

        if score > 0:
            scored.append({
                **doc,
                "score": score,
            })

    scored.sort(
        key=lambda x: (
            x["score"],
            x["source"],
            json.dumps(x["content"], ensure_ascii=False),
        ),
        reverse=True
    )

    return scored[:top_k]
