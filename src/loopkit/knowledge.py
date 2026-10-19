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
    """One slice of a source document.

    ``id`` is the citation marker an answer uses, such as ``p1``. ``source``
    is the file or inline name the slice came from. ``heading`` is the
    markdown heading in force for this slice, or empty when the document has
    none. ``text`` is the slice the model is allowed to use as evidence.
    """

    id: str
    source: str
    heading: str
    text: str


@dataclass(frozen=True)
class Knowledge:
    """The passages a loop answers from, in the order they were read.

    ``warnings`` lists files that were skipped and why. A load can succeed
    and still carry warnings when a directory mixes text files with other
    types, or when a file is empty or too large to read.
    """

    passages: tuple[Passage, ...]
    warnings: tuple[str, ...] = ()

    NO_KNOWLEDGE_WARNING = (
        "No knowledge base was given. The loop runs in open mode: the model answers from "
        "its own knowledge (and web search when search is on), and the audit checks accuracy "
        "instead of grounding. To answer from documents, pass knowledge= (a file, a directory, "
        "or text), or run `loopkit init`."
    )

    def __post_init__(self) -> None:
        """Refuse a knowledge base that cannot be cited.

        A passage with a blank id or blank text is an error, and so are two
        passages that share an id. Ids are what answers cite, so a duplicate
        would make a citation point at more than one slice. An empty passage
        list is allowed: see :meth:`empty`.
        """

        ids = [passage.id for passage in self.passages]
        if any(not passage.id.strip() or not passage.text.strip() for passage in self.passages):
            raise LoopError("Every passage needs an id and text.")
        if len(ids) != len(set(ids)):
            raise LoopError("Passage ids must be unique.")

    @classmethod
    def empty(cls, reason: str | None = None) -> Knowledge:
        """A knowledge base with no passages.

        The loop still runs: the drafter is told there are no passages and
        the default instructions make it say the answer is not available.
        ``reason`` is the warning the caller sees; the default explains that
        no knowledge was given.
        """

        return cls(passages=(), warnings=(reason or cls.NO_KNOWLEDGE_WARNING,))

    @property
    def is_empty(self) -> bool:
        return not self.passages

    @classmethod
    def load(cls, source: KnowledgeLike | None) -> Knowledge:
        """Build a knowledge base from ``source``.

        ``None`` is an empty knowledge base with a warning, see
        :meth:`empty`. A :class:`Knowledge` instance is returned unchanged. A path is read
        from disk. A string that is not a path is kept as the document when
        it contains whitespace, and rejected when it is a single missing
        token such as ``docs``. A dict maps a document name to its text. A
        list is read in order, and each item may itself be any of those forms.

        Files are split on markdown headings and then into overlapping
        chunks. Passage ids are assigned after that split, as ``p1``, ``p2``,
        and so on, in read order.
        """

        if source is None:
            return cls.empty()
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
    """Read one knowledge value into ``documents``.

    Each document is a ``(name, text)`` pair. ``warnings`` gains a line for
    every file that is skipped. A list is walked from first item to last,
    and each item is collected with the same rules as a top-level value.
    Any other type raises ``LoopError``.
    """

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
    """Turn one string into a path read or an inline document.

    Leading and trailing space is ignored. An empty string is an error. If
    the remaining text is an existing file or directory, it is read from
    disk. If it is not a path but it contains whitespace, it is stored as
    inline text, even when the sentence mentions a file path. A single token
    that is not an existing path is an error, so a missing folder name such
    as ``docs`` is not stored as the document itself.
    """

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
    """Store each dict entry as a named document.

    Keys and values must both be strings. The name and the text are stripped,
    and either being empty is an error. An empty dict is an error. The name
    is later used as the passage source, so two entries with the same name
    stay as two documents and are not merged.
    """

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
    """Read a file, or every supported file under a directory.

    Directory entries are visited in sorted order so passage ids stay the
    same from one machine to another. A path that is neither a file nor a
    directory is an error, and so is a directory with no files after hidden
    names are skipped. The source label for a nested file is the directory
    name, a slash, and the path relative to that directory.
    """

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
    """List files under ``directory`` in sorted order.

    Hidden files and hidden directories are skipped. So are the directory
    names in ``SKIP_DIRS``, including ``.git`` and virtualenvs. Those names
    are removed from the walk before it descends, so their contents are never
    considered.
    """

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
    """Read one file into ``documents``, or record why it was skipped.

    Only suffixes in ``TEXT_SUFFIXES`` are read. Another type, an empty
    file, or a file larger than ``MAX_FILE_BYTES`` appends a warning and
    returns, so a directory of mixed files can still load. Text is decoded
    as UTF-8. Bytes that are not valid UTF-8 are replaced rather than
    failing the whole file. The text is stripped before it is stored.
    """

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
    """Store inline text under a generated source name.

    The first inline document is named ``text``. Later ones are ``text-2``,
    ``text-3``, and so on. The number counts names already in ``documents``
    that are ``text`` or start with ``text-``. That name becomes the passage
    source shown with a citation.
    """

    seen = sum(1 for name, _ in documents if name == "text" or name.startswith("text-"))
    name = "text" if seen == 0 else f"text-{seen + 1}"
    documents.append((name, text))


def _chunk_documents(documents: list[tuple[str, str]]) -> list[Passage]:
    """Split each document into passages and number them.

    A document is split on markdown headings first. Each section is then
    split into windows of about ``CHUNK_CHARS`` characters, overlapping by
    ``CHUNK_OVERLAP`` so a sentence on a boundary is not dropped. Ids are
    assigned only after every window exists, in that same order, starting
    at ``p1``. The heading in force for a window is stored on the passage.
    """

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
    """Split markdown into ``(heading, body)`` pairs.

    A line that starts with ``#`` begins a new section. The heading text is
    that line with the hashes and surrounding space removed. Lines inside a
    fenced code block stay in the current section, including a ``#`` comment
    in a code sample. A shebang (``#!``) is also left in the body. Text
    before the first heading uses an empty heading. A document with no
    headings at all returns one pair whose heading is empty.
    """

    heading = ""
    buffer: list[str] = []
    sections: list[tuple[str, str]] = []
    in_fence = False

    def flush() -> None:
        """Save the lines gathered since the previous heading.

        The heading stored with the body is the one in force when those
        lines were read. A section whose body is empty after stripping is
        dropped, so a heading with nothing under it does not become a passage.
        """

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
    """Split ``body`` into chunks near ``size`` characters.

    A body that already fits is returned as one chunk. Otherwise it is
    packed a paragraph at a time, breaking on blank lines. When the next
    paragraph would pass ``size``, the current chunk is closed and the next
    one starts with the last ``overlap`` characters of the previous chunk,
    so a sentence that crosses the cut is still visible. A paragraph longer
    than ``size`` has no blank line to break on, so it is sliced by character
    instead.
    """

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
    """Slice ``text`` into fixed windows of ``size`` characters.

    Used when a body or a paragraph has no blank line to pack on. Each window
    after the first starts ``overlap`` characters before the previous window
    ended. An overlap greater than or equal to ``size`` is treated as no
    overlap, so the start index still moves forward and the loop ends.
    """

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
