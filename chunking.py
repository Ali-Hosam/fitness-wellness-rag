import json
from dataclasses import asdict, dataclass
from pathlib import Path


INPUT_PATH = Path(__file__).with_name("fitness_wellness_corpus_cleaned.txt")
OUTPUT_PATH = Path(__file__).with_name("fitness_wellness_chunks.jsonl")

TARGET_WORDS = 200
MAX_WORDS = 300
OVERLAP_WORDS = 60

CHAPTER_TITLES = {
    "Healthy Behaviors",
    "Fitness Principles",
    "Cardiorespiratory Fitness",
    "Muscular Fitness",
    "Flexibility",
    "Body Composition",
    "Nutrition",
    "Weight Management",
    "Stress",
    "Cardiovascular Disease",
    "Cancer",
    "Substance Use and Abuse",
    "Sexually Transmitted Infections",
}

CHAPTER_ALIASES = {
    "Healthy Behaviors and Wellness": "Healthy Behaviors",
}


@dataclass
class Chunk:
    chunk_id: int
    chapter: str | None
    section: str | None
    text: str
    word_count: int


def classify_heading(paragraph: str) -> tuple[str, str] | None:
    """Separate known chapter titles from likely section headings."""
    normalized = paragraph.strip()
    if normalized in CHAPTER_ALIASES:
        return "chapter", CHAPTER_ALIASES[normalized]
    if normalized in CHAPTER_TITLES:
        return "chapter", normalized

    # Short, punctuation-free lines are likely section labels; bullets remain content.
    words = normalized.split()
    looks_like_section = (
        0 < len(words) <= 12
        and not normalized.startswith(("•", "-", "*"))
        and not normalized.endswith((".", "?", "!", ":", ";"))
    )
    if looks_like_section:
        return "section", normalized
    return None


def word_count(text: str) -> int:
    return len(text.split())


def build_chunk(paragraphs: list[str], chapter: str | None, section: str | None) -> Chunk:
    text = "\n\n".join(paragraphs)
    return Chunk(
        chunk_id=0,
        chapter=chapter,
        section=section,
        text=text,
        word_count=word_count(text),
    )


def chunk_text(text: str) -> list[Chunk]:
    paragraphs = [paragraph.strip() for paragraph in text.split("\n\n") if paragraph.strip()]
    chunks = []
    current_paragraphs = []
    current_chapter = None
    current_section = None

    for paragraph in paragraphs:
        heading = classify_heading(paragraph)

        if heading and current_paragraphs:
            # A heading belongs with the text after it, not the previous section.
            chunks.append(build_chunk(current_paragraphs, current_chapter, current_section))
            overlap = []
            overlap_words = 0
            for previous in reversed(current_paragraphs):
                previous_words = word_count(previous)
                overlap.insert(0, previous)
                overlap_words += previous_words
                if overlap_words >= OVERLAP_WORDS:
                    break
            current_paragraphs = overlap

        if heading:
            heading_type, heading_value = heading
            if heading_type == "chapter":
                current_chapter = heading_value
                current_section = None
            else:
                current_section = heading_value

        candidate = current_paragraphs + [paragraph]
        candidate_words = word_count("\n\n".join(candidate))

        # Keep complete paragraphs together until the target size is reached.
        if current_paragraphs and candidate_words > TARGET_WORDS:
            chunks.append(build_chunk(current_paragraphs, current_chapter, current_section))

            # Overlap repeats complete trailing paragraphs, never a partial sentence.
            overlap = []
            overlap_words = 0
            for previous in reversed(current_paragraphs):
                previous_words = word_count(previous)
                overlap.insert(0, previous)
                overlap_words += previous_words
                if overlap_words >= OVERLAP_WORDS:
                    break
            current_paragraphs = overlap

            # A full paragraph plus overlap must still respect the hard maximum.
            if word_count("\n\n".join(current_paragraphs + [paragraph])) > MAX_WORDS:
                current_paragraphs = []

        current_paragraphs.append(paragraph)

        # A single unusually long paragraph is kept intact rather than damaged.
        if word_count("\n\n".join(current_paragraphs)) >= MAX_WORDS:
            chunks.append(build_chunk(current_paragraphs, current_chapter, current_section))
            current_paragraphs = []

    if current_paragraphs:
        chunks.append(build_chunk(current_paragraphs, current_chapter, current_section))

    for chunk_id, chunk in enumerate(chunks):
        chunk.chunk_id = chunk_id
    return chunks


def write_chunks(chunks: list[Chunk], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8") as output_file:
        for chunk in chunks:
            output_file.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")


if __name__ == "__main__":
    chunks = chunk_text(INPUT_PATH.read_text(encoding="utf-8"))
    write_chunks(chunks, OUTPUT_PATH)
    print(f"Wrote {len(chunks)} chunks to {OUTPUT_PATH}")