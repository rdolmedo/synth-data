"""Physical LF records for JSONL, including literal Unicode separators in strings."""

def physical_lines(text: str) -> list[str]:
    # Remove only the file's final LF, not a real blank record. CRLF is a line
    # ending; U+0085/U+2028 and other Unicode separators are JSON string data.
    if not text:
        return []
    return [line.removesuffix('\r') for line in text.removesuffix('\n').split('\n')]
