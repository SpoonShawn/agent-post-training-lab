import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KNOWLEDGE_DIR = ROOT / "data" / "knowledge"


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


def search_knowledge(query: str, top_k: int = 3):
    query_words = set(query.lower().split())
    scored = []

    for doc in load_docs():
        text = json.dumps(doc["content"], ensure_ascii=False).lower()

        score = sum(
            word in text
            for word in query_words
        )

        if score > 0:
            scored.append({
                **doc,
                "score": score
            })

    scored.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    return scored[:top_k]
