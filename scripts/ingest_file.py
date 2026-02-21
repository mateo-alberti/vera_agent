from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Iterable, List

# Ensure repo root is on sys.path for app imports when running from scripts/.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.domain.ports import VectorEmbeddingItem
from app.infrastructure.chroma_vector_store import get_chroma_vector_store
from app.infrastructure.openai_adapter import OpenAIAdapter


@dataclass(frozen=True)
class IngestArgs:
    file_path: Path
    source: str
    name: str
    min_length: int


def parse_args() -> IngestArgs:
    parser = argparse.ArgumentParser(
        description="Ingest a text file into Chroma by paragraph."
    )
    parser.add_argument("--file", required=True, help="Path to the input file.")
    parser.add_argument("--source", required=True, help="Source tag (e.g., 'manual', 'docs').")
    parser.add_argument("--name", required=True, help="Logical file name for IDs/metadata.")
    parser.add_argument(
        "--min-length",
        type=int,
        default=20,
        help="Minimum paragraph length to include (default: 20).",
    )
    args = parser.parse_args()
    return IngestArgs(
        file_path=Path(args.file),
        source=args.source.strip(),
        name=args.name.strip(),
        min_length=args.min_length,
    )


def read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def read_text(path: Path) -> str:
    if path.suffix.lower() != ".txt":
        raise ValueError(f"Unsupported file type: {path.suffix}. Only .txt files are supported.")
    return read_text_file(path)


def split_paragraphs(text: str) -> List[str]:
    paragraphs = [p.strip() for p in text.split("\n\n")]
    return [p for p in paragraphs if p]


def build_items(
    paragraphs: Iterable[str], *, args: IngestArgs
) -> List[VectorEmbeddingItem]:
    paragraphs_list = [p for p in paragraphs if len(p) >= args.min_length]
    if not paragraphs_list:
        return []

    adapter = OpenAIAdapter()
    embeddings = adapter.embed_texts(paragraphs_list).vectors

    items: List[VectorEmbeddingItem] = []
    for idx, paragraph in enumerate(paragraphs_list, start=1):
        item_id = f"{args.name}:{idx}"
        items.append(
            VectorEmbeddingItem(
                id=item_id,
                embedding=embeddings[idx - 1],
                document=paragraph,
                metadata={
                    "source": args.source,
                    "file_name": args.name,
                    "paragraph_index": idx,
                    "file_path": str(args.file_path),
                },
            )
        )
    return items


def ingest_file(args: IngestArgs) -> int:
    if not args.file_path.exists():
        raise FileNotFoundError(f"File not found: {args.file_path}")

    text = read_text(args.file_path)
    paragraphs = split_paragraphs(text)
    items = build_items(paragraphs, args=args)

    if not items:
        return 0

    store = get_chroma_vector_store()
    return store.upsert_embeddings(items)


def main() -> None:
    args = parse_args()
    inserted = ingest_file(args)
    print(f"Inserted {inserted} chunks into Chroma")


if __name__ == "__main__":
    main()
