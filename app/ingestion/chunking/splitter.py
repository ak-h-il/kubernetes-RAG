from typing import List

import logfire


def chunk_text(text: str, chunk_size: int = 1500, chunk_overlap: int = 150) -> List[str]:
    """
    Semantic chunker that hierarchically splits by paragraphs (\n\n),
    lines (\n), sentences, and words, ensuring chunks stay within chunk_size
    while maintaining a sliding context overlap.
    """
    with logfire.span("✂️ Text Chunking", text_length=len(text)):
        if not text or not text.strip():
            return []

        def _split_into_blocks(content: str, separators: list[str]) -> list[str]:
            if not separators:
                step = max(1, chunk_size - chunk_overlap)
                return [content[i : i + chunk_size] for i in range(0, len(content), step)] if content else []

            sep = separators[0]
            remaining = separators[1:]
            parts = content.split(sep)
            blocks = []
            for part in parts:
                part = part.strip()
                if not part:
                    continue
                if len(part) <= chunk_size:
                    blocks.append(part)
                else:
                    blocks.extend(_split_into_blocks(part, remaining))
            return blocks

        blocks = _split_into_blocks(text, ["\n\n", "\n", ". ", " "])
        chunks = []
        current_chunk = ""

        for block in blocks:
            if not current_chunk:
                current_chunk = block
            elif len(current_chunk) + 2 + len(block) <= chunk_size:
                current_chunk += "\n\n" + block
            else:
                chunks.append(current_chunk.strip())
                if chunk_overlap > 0 and len(current_chunk) > chunk_overlap:
                    overlap = current_chunk[-chunk_overlap:].strip()
                    current_chunk = overlap + "\n\n" + block
                else:
                    current_chunk = block

        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        valid_chunks = [c for c in chunks if c.strip()]
        logfire.info(f"✅ Generated {len(valid_chunks)} chunks")
        return valid_chunks
