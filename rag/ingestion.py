from pathlib import Path


def read_text(path):
    p = Path(path)
    if p.suffix.lower() == ".pdf":
        from pypdf import PdfReader           # optional dependency
        return "\n".join(pg.extract_text() or "" for pg in PdfReader(str(p)).pages)
    return p.read_text(encoding="utf-8", errors="ignore")


def chunk_text(text, size=700, overlap=100):
    """Paragraph-aware chunks of roughly `size` characters."""
    paras, chunks, cur = [t.strip() for t in text.split("\n\n") if t.strip()], [], ""
    for p in paras:
        if len(cur) + len(p) + 2 <= size:
            cur = f"{cur}\n\n{p}".strip()
        else:
            if cur:
                chunks.append(cur)
            cur = (cur[-overlap:] + "\n\n" + p) if cur and len(p) < size else p
    if cur:
        chunks.append(cur)
    return chunks


def load_corpus(folder):
    docs = []
    for p in sorted(Path(folder).rglob("*")):
        if p.suffix.lower() in {".md", ".txt", ".pdf"}:
            for i, c in enumerate(chunk_text(read_text(p))):
                docs.append({"id": f"{p.stem}#{i}", "source": p.name, "text": c})
    return docs


def make_docs(name, text):
    return [{"id": f"{name}#{i}", "source": name, "text": c} for i, c in enumerate(chunk_text(text))]
