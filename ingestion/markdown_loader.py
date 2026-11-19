"""Markdown-aware loaders: PDF (pymupdf4llm) and HTML (cleaned, then markdownify)."""
from pathlib import Path

def load_pdf(path) -> str:
    import pymupdf4llm
    return pymupdf4llm.to_markdown(str(path))

def load_html(path) -> str:
    from bs4 import BeautifulSoup
    from markdownify import markdownify as md
    soup = BeautifulSoup(Path(path).read_text(encoding="utf-8"), "html.parser")
    for t in soup(["script", "style", "nav", "footer", "header", "noscript"]): t.decompose()
    return md(str(soup.body or soup), heading_style="ATX").strip()

def load_document(path) -> str:
    p = Path(path)
    if p.suffix.lower() == ".pdf": return load_pdf(p)
    if p.suffix.lower() in (".html", ".htm"): return load_html(p)
    return p.read_text(encoding="utf-8")
