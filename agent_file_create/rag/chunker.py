import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    chunk_index: int
    section_path: str
    content: str
    parent_chunk_id: str = ""  # ID of the parent (larger) chunk for context-window retrieval


def _normalize_text(text: str) -> str:
    s = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"[ \t]+\n", "\n", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def _split_md_sections(text: str) -> list[tuple[str, str]]:
    lines = (text or "").splitlines()
    out: list[tuple[str, str]] = []
    buf: list[str] = []
    stack: list[tuple[int, str]] = []
    cur_path = ""

    def flush():
        nonlocal buf, cur_path
        body = "\n".join(buf).strip()
        if cur_path or body:
            out.append((cur_path.strip(" /"), body))
        buf = []

    for line in lines:
        m = re.match(r"^\s*(#{1,6})\s+(.+?)\s*$", line)
        if m:
            flush()
            lvl = len(m.group(1))
            title = m.group(2).strip()
            while stack and stack[-1][0] >= lvl:
                stack.pop()
            stack.append((lvl, title))
            cur_path = " / ".join([t for _, t in stack if t]).strip(" /")
            continue
        buf.append(line)

    flush()
    return out


def _is_likely_title(text: str) -> bool:
    """Heuristic: short line, no trailing punctuation, reasonable alpha ratio."""
    s = str(text or "").strip()
    n = len(s)
    if n < 3 or n > 30:
        return False
    if re.search(r"[。！？!?，,、；;：:.]$", s):
        return False
    # Should not be purely numeric/punctuation
    alpha = len(re.findall(r"[\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaffA-Za-z0-9]", s))
    if alpha / max(n, 1) < 0.5:
        return False
    return True


def _split_paragraphs(text: str) -> list[str]:
    blocks = [x.strip() for x in re.split(r"\n\s*\n", text or "") if x.strip()]
    out: list[str] = []
    for b in blocks:
        b = re.sub(r"\s+", " ", b).strip()
        if b:
            out.append(b)
    return out


def _split_md_blocks(text: str) -> list[str]:
    lines = (text or "").splitlines()
    out: list[str] = []
    buf: list[str] = []
    in_code = False
    code_fence = ""

    def flush_buf():
        nonlocal buf
        if not buf:
            return
        cleaned: list[str] = []
        for ln in buf:
            if ln.strip():
                cleaned.append(re.sub(r"[ \t]+", " ", ln).rstrip())
            else:
                cleaned.append("")
        s = "\n".join(cleaned).strip("\n").strip()
        if s:
            out.append(s)
        buf = []

    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        s = line.lstrip()
        if s.startswith("```"):
            if not in_code:
                flush_buf()
                in_code = True
                code_fence = s[:3]
                buf = [line]
            else:
                buf.append(line)
                flush_buf()
                in_code = False
                code_fence = ""
            i += 1
            continue

        if in_code:
            buf.append(line)
            i += 1
            continue

        if "|" in line and line.strip():
            nxt = lines[i + 1] if i + 1 < n else ""
            if ("|" in nxt and nxt.strip()) or re.match(r"^\s*\|?[\s:-]+\|[\s|:-]*\s*$", nxt or ""):
                flush_buf()
                tbuf: list[str] = []
                j = i
                while j < n and lines[j].strip() and "|" in lines[j]:
                    tbuf.append(lines[j].rstrip())
                    j += 1
                s2 = "\n".join(tbuf).strip()
                if s2:
                    out.append(s2)
                i = j
                continue

        if not line.strip():
            flush_buf()
            i += 1
            continue

        buf.append(line)
        i += 1

    flush_buf()
    return out


# Separator hierarchy for recursive splitting: coarse → fine
_RECURSIVE_SEPARATORS = [
    "\n\n",
    "\n",
    r"(?<=[。！？!?])(?=\S)",
    r"(?<=[.!?])\s+(?=\S)",
    r"(?<=[；;])(?=\S)",
    r"(?<=[，,])(?=\S)",
    " ",
]


def _recursive_split_text(text: str, target_chars: int, separators: list[str] | None = None) -> list[str]:
    """Split an oversized text block by trying ever-finer separators recursively."""
    if separators is None:
        separators = _RECURSIVE_SEPARATORS

    s = str(text or "").strip()
    if not s or len(s) <= target_chars:
        return [s] if s else []

    if not separators:
        return [s[i : i + target_chars] for i in range(0, len(s), target_chars)]

    sep = separators[0]
    rest = separators[1:]

    parts = re.split(sep, s)
    parts = [p.strip() for p in parts if p.strip()]

    if len(parts) <= 1:
        return _recursive_split_text(s, target_chars, rest)

    result: list[str] = []
    buf = ""
    for part in parts:
        candidate = (buf + "\n" + part).strip() if buf else part
        if len(candidate) <= target_chars:
            buf = candidate
        else:
            if buf:
                result.append(buf)
            if len(part) > target_chars:
                # Only recurse if there's meaningful room to split; otherwise keep slightly-oversized intact
                if rest and len(part) <= target_chars * 1.3:
                    if buf:
                        result.append(buf)
                    result.append(part)
                    buf = ""
                else:
                    for sp in _recursive_split_text(part, target_chars, rest):
                        candidate2 = (buf + "\n" + sp).strip() if buf else sp
                        if len(candidate2) <= target_chars:
                            buf = candidate2
                        else:
                            if buf:
                                result.append(buf)
                            buf = sp
            else:
                buf = part
    if buf:
        result.append(buf)
    return result


def _overlap_tail(text: str, *, overlap_chars: int) -> str:
    s = str(text or "").strip()
    overlap = max(0, int(overlap_chars or 0))
    if not s or overlap <= 0:
        return ""
    if len(s) <= overlap:
        return s
    # Search backward from the target position for the strongest semantic boundary.
    # This ensures overlap always starts at a clean sentence / clause edge,
    # rather than in the middle of a word or number.
    target = max(0, len(s) - overlap)
    for sep in ["\n\n", "\n", "。", "！", "？", "!", "?", "；", ";"]:
        j = s.rfind(sep, max(0, target - overlap // 3), target + len(sep))
        if j >= 0:
            return s[j + len(sep):].strip()
    return s[target:].strip()


def _chunk_by_size(paragraphs: list[str], *, target_chars: int, overlap_chars: int) -> list[str]:
    target = max(200, int(target_chars or 0))
    overlap = max(0, int(overlap_chars or 0))
    out: list[str] = []
    cur: list[str] = []
    cur_len = 0
    for p0 in paragraphs:
        parts = [p0]
        if len(p0) > target:
            parts = _recursive_split_text(p0, target)
        for p in parts:
            add_len = len(p) + (1 if cur else 0)
            if cur and cur_len + add_len > target:
                text = "\n".join(cur).strip()
                if text:
                    out.append(text)
                if overlap > 0:
                    tail = _overlap_tail(text, overlap_chars=overlap)
                    cur = [tail] if tail.strip() else []
                    cur_len = len(tail) if cur else 0
                else:
                    cur = []
                    cur_len = 0
            cur.append(p)
            cur_len += add_len
    tail = "\n".join(cur).strip()
    if tail:
        out.append(tail)
    return out


def chunk_text(
    *,
    doc_id: str,
    text: str,
    title: str = "",
    is_markdown: bool = True,
    target_chars: int = 1200,
    overlap_chars: int = 120,
    max_chunks: int = 800,
) -> list[Chunk]:
    doc_id = str(doc_id or "").strip() or "doc"
    title = str(title or "").strip()
    t = _normalize_text(text)
    if not t:
        return []

    sections: list[tuple[str, str]]
    if is_markdown:
        sections = _split_md_sections(t)
    else:
        # For non-markdown text, detect titles to create coarse sections
        # This mirrors Chatchat's zh_title_enhance concept
        lines = [l.strip() for l in t.splitlines()]
        detected: list[tuple[str, str]] = []
        cur_title = title or ""
        cur_lines: list[str] = []
        for line in lines:
            if _is_likely_title(line):
                body = "\n".join(cur_lines).strip()
                if body:
                    detected.append((cur_title, body))
                cur_title = line
                cur_lines = []
            else:
                cur_lines.append(line)
        body = "\n".join(cur_lines).strip()
        if body:
            detected.append((cur_title, body))
        sections = detected if detected else [(title, t)]

    chunks: list[Chunk] = []
    idx = 0
    for section_path, body in sections:
        section_path = str(section_path or "").strip()
        if title and (not section_path):
            section_path = title
        if title and section_path:
            parts = [x.strip() for x in section_path.split("/") if x.strip()]
            if not parts:
                parts = [title]
            elif parts[0] != title:
                parts = [title, *parts]
            deduped: list[str] = []
            for p in parts:
                if deduped and deduped[-1] == p:
                    continue
                deduped.append(p)
            section_path = " / ".join(deduped)
        if not body:
            continue
        paras = _split_md_blocks(body) if is_markdown else _split_paragraphs(body)
        if not paras:
            continue
        parts = _chunk_by_size(paras, target_chars=target_chars, overlap_chars=overlap_chars)
        for part in parts:
            header = section_path.strip()
            content = (header + "\n" + part).strip() if header else part.strip()
            if not content:
                continue
            cid = f"{doc_id}:{idx}"
            chunks.append(Chunk(chunk_id=cid, doc_id=doc_id, chunk_index=idx, section_path=section_path, content=content))
            idx += 1
            if len(chunks) >= int(max_chunks or 0):
                # Assign parent IDs to existing chunks before returning
                _assign_parent_chunks(chunks, doc_id, parent_size=4)
                return chunks
    # Assign parent chunks: group every parent_size consecutive children
    _assign_parent_chunks(chunks, doc_id, parent_size=4)
    return chunks


def _assign_parent_chunks(chunks: list[Chunk], doc_id: str, parent_size: int = 4) -> None:
    """Assign parent_chunk_id by grouping consecutive child chunks.

    Parent chunks are used for context-window retrieval: when a child chunk
    matches a query, its parent (larger context) can also be returned.
    """
    n = len(chunks)
    if n <= parent_size:
        return
    parent_idx = 0
    i = 0
    while i < n:
        group = chunks[i : i + parent_size]
        pid = f"{doc_id}:parent:{parent_idx}"
        for ch in group:
            # Since Chunk is frozen, replace it with a new instance
            chunks[chunks.index(ch)] = Chunk(
                chunk_id=ch.chunk_id,
                doc_id=ch.doc_id,
                chunk_index=ch.chunk_index,
                section_path=ch.section_path,
                content=ch.content,
                parent_chunk_id=pid,
            )
        parent_idx += 1
        i += parent_size
