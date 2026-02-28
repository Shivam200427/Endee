import os
import re
from PyPDF2 import PdfReader


def extract_text_from_pdf(file_path):
    """Read a PDF file and return the full text content."""
    reader = PdfReader(file_path)
    pages_text = []
    for page_num, page in enumerate(reader.pages):
        text = page.extract_text()
        if text and text.strip():
            pages_text.append(text.strip())
    return "\n\n".join(pages_text)


def extract_text_from_uploaded(uploaded_file):
    """Handle a Streamlit UploadedFile object and extract text."""
    reader = PdfReader(uploaded_file)
    pages_text = []
    for page in reader.pages:
        text = page.extract_text()
        if text and text.strip():
            pages_text.append(text.strip())
    return "\n\n".join(pages_text)


# Common section headers found in resumes, reports, etc.
_SECTION_HEADERS = [
    "objective", "summary", "skills", "certifications", "courses",
    "achievements", "education", "experience", "projects",
    "work experience", "professional experience", "technical skills",
    "publications", "awards", "interests", "hobbies", "references",
    "contact", "about", "introduction", "background", "methodology",
    "results", "conclusion", "abstract", "overview",
]


def _detect_sections(text):
    """
    Split text into labelled sections based on common document headers.

    Returns a list of (section_label, section_body) tuples.  If a block
    has no recognisable header it gets the label from the previous section,
    or an empty string for the very first block.
    """
    text = re.sub(r'\n{3,}', '\n\n', text)

    # Build pattern that matches a standalone header line
    header_pat = re.compile(
        r'^(' + '|'.join(re.escape(h) for h in _SECTION_HEADERS) + r')\s*$',
        re.IGNORECASE | re.MULTILINE,
    )

    sections = []
    current_label = ""
    current_lines = []

    for line in text.split('\n'):
        stripped = line.strip()
        m = header_pat.match(stripped)
        if m:
            # Flush accumulated lines under the previous label
            body = '\n'.join(current_lines).strip()
            if body:
                sections.append((current_label, body))
            current_label = m.group(1).strip().title()
            current_lines = []
        else:
            current_lines.append(line)

    # Flush last section
    body = '\n'.join(current_lines).strip()
    if body:
        sections.append((current_label, body))

    # Auto-label the first unlabeled block as "Contact" if it looks like
    # contact information (has email, phone, or @ patterns)
    if sections and not sections[0][0]:
        first_body = sections[0][1]
        has_email = bool(re.search(r'[\w.+-]+@[\w-]+\.[\w.]+', first_body))
        has_phone = bool(re.search(r'\+?\d[\d\s\-]{7,}', first_body))
        if has_email or has_phone:
            sections[0] = ("Contact", first_body)

    return sections


def _split_into_segments(text_block):
    """
    Break a text block into smaller segments on paragraph breaks,
    bullet points, or sentence boundaries when paragraphs are long.
    """
    paragraphs = re.split(r'\n\n+', text_block)

    segments = []
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        if len(para) <= 500:
            segments.append(para)
        else:
            # Split on bullets, numbered items, or sentence endings
            parts = re.split(r'\n(?=•|\-\s|\d+\.)|(?<=[.!?])\s+(?=[A-Z])', para)
            for part in parts:
                part = part.strip()
                if part:
                    segments.append(part)

    return segments


def _normalize_for_dedup(text):
    """Collapse whitespace so near-duplicate chunks can be caught."""
    return re.sub(r'\s+', ' ', text).strip().lower()


def chunk_text(text, chunk_size=400, overlap=0):
    """
    Split document text into context-rich, deduplicated chunks.

    Detects document sections (e.g. Skills, Projects, Education) and
    prepends the section label to each chunk so the embedding model
    captures the semantic role of the content—not just the words.
    Overlap is set to 0 by default to avoid near-duplicate vectors.
    """
    if not text or not text.strip():
        return []

    sections = _detect_sections(text)
    if not sections:
        return []

    chunks = []
    seen_normalized = set()

    for label, body in sections:
        segments = _split_into_segments(body)
        if not segments:
            continue

        current_parts = []
        current_len = 0

        for seg in segments:
            seg_len = len(seg)

            # If a single segment exceeds chunk_size, break it by lines
            if seg_len > chunk_size:
                # Flush first
                if current_parts:
                    _flush_chunk(chunks, seen_normalized, current_parts, label)
                    current_parts, current_len = [], 0

                sub_parts, sub_len = [], 0
                for line in seg.split('\n'):
                    line = line.strip()
                    if not line:
                        continue
                    if sub_len + len(line) > chunk_size and sub_parts:
                        _flush_chunk(chunks, seen_normalized, sub_parts, label)
                        sub_parts, sub_len = [], 0
                    sub_parts.append(line)
                    sub_len += len(line) + 1

                if sub_parts:
                    current_parts, current_len = sub_parts, sub_len
                continue

            # Would adding this segment exceed the limit?
            if current_len + seg_len > chunk_size and current_parts:
                _flush_chunk(chunks, seen_normalized, current_parts, label)

                # Optional overlap: carry the last segment for continuity
                if overlap > 0 and len(current_parts) >= overlap:
                    carry = current_parts[-overlap:]
                    current_parts = list(carry)
                    current_len = sum(len(s) for s in carry) + len(carry) - 1
                else:
                    current_parts, current_len = [], 0

            current_parts.append(seg)
            current_len += seg_len + 1

        # Flush whatever remains in this section
        if current_parts:
            _flush_chunk(chunks, seen_normalized, current_parts, label)

    return chunks


def _flush_chunk(chunks, seen_normalized, parts, section_label):
    """
    Join parts, prepend the section label, and append to chunks
    if the resulting text is not a duplicate.
    """
    body = "\n".join(parts).strip()
    if not body:
        return

    # Prepend section label for embedding context
    if section_label:
        text = f"[{section_label}] {body}"
    else:
        text = body

    norm = _normalize_for_dedup(text)
    if norm in seen_normalized:
        return

    seen_normalized.add(norm)
    chunks.append({"id": f"chunk_{len(chunks)}", "text": text})


def save_uploaded_pdf(uploaded_file, save_dir="data"):
    """Save an uploaded PDF to the data directory and return the path."""
    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path
