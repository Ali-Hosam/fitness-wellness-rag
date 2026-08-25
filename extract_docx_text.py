from pathlib import Path
import re
from zipfile import ZipFile
from xml.etree import ElementTree


DOCX_PATH = Path(__file__).with_name("ALG Concepts of Fitness Wellness.docx")
TEXT_PATH = Path(__file__).with_name("fitness_wellness_corpus.txt")
CLEANED_TEXT_PATH = Path(__file__).with_name("fitness_wellness_corpus_cleaned.txt")
WORD_NAMESPACE = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"


def extract_text(docx_path: Path) -> str:
    with ZipFile(docx_path) as archive:
        document = ElementTree.fromstring(archive.read("word/document.xml"))

    paragraphs = []
    for element in document.iter():
        if element.tag == f"{{{WORD_NAMESPACE}}}p":
            text = "".join(
                node.text or ""
                for node in element.iter()
                if node.tag == f"{{{WORD_NAMESPACE}}}t"
            ).strip()
            if text:
                paragraphs.append(text)

    return "\n\n".join(paragraphs) + "\n"


def clean_text(text: str) -> str:
    paragraphs = [paragraph.strip() for paragraph in text.split("\n\n")]

    # Page labels are layout noise from the textbook export, not corpus content.
    paragraphs = [
        paragraph
        for paragraph in paragraphs
        if not re.fullmatch(r"Page\s+\d+", paragraph, flags=re.IGNORECASE)
    ]

    # The opening publishing information is not useful for answering textbook questions.
    first_chapter = next(
        (
            index
            for index, paragraph in enumerate(paragraphs)
            if paragraph == "Healthy Behaviors and Wellness"
        ),
        0,
    )
    paragraphs = paragraphs[first_chapter:]

    # Exported headings can be duplicated on adjacent lines; preserve one copy.
    deduplicated = []
    for paragraph in paragraphs:
        if not deduplicated or paragraph != deduplicated[-1]:
            deduplicated.append(paragraph)

    return "\n\n".join(deduplicated) + "\n"


if __name__ == "__main__":
    raw_text = extract_text(DOCX_PATH)
    TEXT_PATH.write_text(raw_text, encoding="utf-8")
    CLEANED_TEXT_PATH.write_text(clean_text(raw_text), encoding="utf-8")
    print(f"Extracted text to {TEXT_PATH}")
    print(f"Cleaned text to {CLEANED_TEXT_PATH}")