"""Load text and split it into stable passages.

A string that is an existing path is read from disk. Any other string that
contains whitespace is the document itself, even if it mentions a file path.
A single token that is not an existing path is an error, so a missing ``docs``
folder is not silently treated as the knowledge base.
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from loopkit.errors import LoopError

TEXT_SUFFIXES = {
    ".md",
    ".markdown",
    ".txt",
    ".rst",
    ".text",
    ".json",
    ".yaml",
    ".yml",
    ".toml",
    ".csv",
}
SKIP_DIRS = {
    ".git",
    ".hg",
    ".svn",
    "node_modules",
    "__pycache__",
    ".venv",
    "venv",
    "dist",
    "build",
    ".mypy_cache",
    ".pytest_cache",
}
MAX_FILE_BYTES = 1_000_000
CHUNK_CHARS = 1400
CHUNK_OVERLAP = 200

@dataclass(frozen=True)
class Passage:
    """One slice of a source document. ``id`` is what answers cite."""

    id: str
    source: str
    heading: str
    text: str


@dataclass(frozen=True)
class Knowledge:
    """Passages in read order, plus notes about files that were skipped."""

    passages: tuple[Passage, ...]
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.passages:
            raise LoopError("The knowledge base is empty.")
        ids = [passage.id for passage in self.passages]
        if any(not passage.id.strip() or not passage.text.strip() for passage in self.passages):
            raise LoopError("Every passage needs an id and text.")
        if len(ids) != len(set(ids)):
            raise LoopError("Passage ids must be unique.")

    @classmethod
    def load(cls, source: KnowledgeLike) -> Knowledge:
        if isinstance(source, Knowledge):
            return source
        documents: list[tuple[str, str]] = []
        warnings: list[str] = []
        _collect(source, documents, warnings)
        if not documents:
            detail = " ".join(warnings)
            suffix = f" {detail}" if detail else ""
            raise LoopError(f"No readable text in the knowledge base.{suffix}")
        passages = _chunk_documents(documents)
        if not passages:
            raise LoopError("The knowledge base has no text to read.")
        return cls(passages=tuple(passages), warnings=tuple(warnings))


KnowledgeLike = str | Path | Knowledge | Mapping[str, str] | Sequence[object]


def _collect(source: object, documents: list[tuple[str, str]], warnings: list[str]) -> None:
    if isinstance(source, Path):
        _collect_path(source.expanduser(), documents, warnings)
        return
    if isinstance(source, str):
        _collect_string(source, documents, warnings)
        return
    if isinstance(source, Mapping):
        _collect_mapping(source, documents)
        return
    if isinstance(source, Sequence) and not isinstance(source, (bytes, bytearray)):
        if not source:
            raise LoopError("The knowledge list is empty.")
        for item in source:
            _collect(item, documents, warnings)
        return
    raise LoopError(
        "Knowledge must be a path, a string of text, a dict of name to text, or a list of those."
    )


def _collect_string(value: str, documents: list[tuple[str, str]], warnings: list[str]) -> None:
    stripped = value.strip()
    if not stripped:
        raise LoopError("Knowledge text is empty.")
    path = Path(stripped).expanduser()
    if path.exists():
        _collect_path(path, documents, warnings)
        return
    if any(character.isspace() for character in stripped):
        _add_inline(documents, stripped)
        return
    raise LoopError(
        f"No knowledge at {stripped}. Pass a file or directory that exists, "
        "or pass the source text as a sentence."
    )


def _collect_mapping(source: Mapping[object, object], documents: list[tuple[str, str]]) -> None:
    if not source:
        raise LoopError("The knowledge dict is empty.")
    for key, value in source.items():
        if not isinstance(key, str) or not isinstance(value, str):
            raise LoopError("Inline knowledge must be a dict of name to text.")
        name = key.strip()
        text = value.strip()
        if not name or not text:
            raise LoopError("Inline knowledge names and text must not be empty.")
        documents.append((name, text))


def _collect_path(path: Path, documents: list[tuple[str, str]], warnings: list[str]) -> None:
    if path.is_file():
        _read_file(path, path.name, documents, warnings)
        return
    if not path.is_dir():
        raise LoopError(f"No knowledge at {path}.")
    files = _iter_files(path)
    if not files:
        raise LoopError(f"No files in {path}.")
    label = path.name or str(path)
    for file in files:
        relative = file.relative_to(path).as_posix()
        _read_file(file, f"{label}/{relative}", documents, warnings)


def _iter_files(directory: Path) -> list[Path]:
    files: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(directory):
        dirnames[:] = sorted(
            name for name in dirnames if name not in SKIP_DIRS and not name.startswith(".")
        )
        for name in sorted(filenames):
            if name.startswith("."):
                continue
            candidate = Path(dirpath) / name
            if candidate.is_file():
                files.append(candidate)
    return files


def _read_file(
    path: Path,
    source: str,
    documents: list[tuple[str, str]],
    warnings: list[str],
) -> None:
    suffix = path.suffix.lower()
    if suffix not in TEXT_SUFFIXES:
        kind = suffix or "no suffix"
        warnings.append(f"Skipped {source}: unsupported file type ({kind}).")
        return
    try:
        size = path.stat().st_size
    except OSError as exc:
        warnings.append(f"Skipped {source}: {exc}.")
        return
    if size > MAX_FILE_BYTES:
        warnings.append(f"Skipped {source}: larger than {MAX_FILE_BYTES} bytes.")
        return
    if size == 0:
        warnings.append(f"Skipped {source}: empty file.")
        return
    try:
        text = path.read_text(encoding="utf-8", errors="replace").strip()
    except OSError as exc:
        warnings.append(f"Skipped {source}: {exc}.")
        return
    if not text:
        warnings.append(f"Skipped {source}: empty file.")
        return
    documents.append((source, text))


def _add_inline(documents: list[tuple[str, str]], text: str) -> None:
    seen = sum(1 for name, _ in documents if name == "text" or name.startswith("text-"))
    name = "text" if seen == 0 else f"text-{seen + 1}"
    documents.append((name, text))


def _chunk_documents(documents: list[tuple[str, str]]) -> list[Passage]:
    raw: list[tuple[str, str, str]] = []
    for source, text in documents:
        for heading, body in _sections(text):
            for piece in _windows(body, CHUNK_CHARS, CHUNK_OVERLAP):
                raw.append((source, heading, piece))
    return [
        Passage(id=f"p{index}", source=source, heading=heading, text=text)
        for index, (source, heading, text) in enumerate(raw, start=1)
    ]


def _sections(text: str) -> list[tuple[str, str]]:
    """Split markdown on ATX headings, leaving fenced code blocks intact."""

    heading = ""
    buffer: list[str] = []
    sections: list[tuple[str, str]] = []
    in_fence = False

    def flush() -> None:
        body = "\n".join(buffer).strip()
        if body:
            sections.append((heading, body))

    for line in text.splitlines():
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
            buffer.append(line)
            continue
        if not in_fence and line.startswith("#") and not line.startswith("#!"):
            flush()
            buffer = []
            heading = line.lstrip("#").strip()
            continue
        buffer.append(line)
    flush()
    if not sections and text.strip():
        return [("", text.strip())]
    return sections


def _windows(body: str, size: int, overlap: int) -> list[str]:
    body = body.strip()
    if not body:
        return []
    if len(body) <= size:
        return [body]
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", body) if part.strip()]
    if len(paragraphs) <= 1:
        return _hard_windows(body, size, overlap)

    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        if len(paragraph) > size:
            if current:
                chunks.append(current)
                current = ""
            chunks.extend(_hard_windows(paragraph, size, overlap))
            continue
        if not current:
            current = paragraph
            continue
        candidate = f"{current}\n\n{paragraph}"
        if len(candidate) <= size:
            current = candidate
            continue
        chunks.append(current)
        tail = current[-overlap:].strip()
        overlapped = f"{tail}\n\n{paragraph}" if tail else paragraph
        current = overlapped if len(overlapped) <= size else paragraph
    if current:
        chunks.append(current)
    return [chunk for chunk in chunks if chunk.strip()]


def _hard_windows(text: str, size: int, overlap: int) -> list[str]:
    if overlap >= size:
        overlap = 0
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + size)
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= len(text):
            break
        next_start = max(end - overlap, start + 1)
        start = next_start
    return chunks
