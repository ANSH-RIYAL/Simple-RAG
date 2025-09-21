import re
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class Chunk:
    id: str
    text: str
    heading_path: List[str]
    metadata: dict


def normalize_text(text: str) -> str:
    """Basic text cleanup: dehyphenation, whitespace normalization."""
    # Remove hyphenation across line breaks
    text = re.sub(r'-\s*\n\s*', '', text)
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text


def detect_headings(lines: List[str]) -> List[Tuple[int, int, str]]:
    """Detect headings and return (line_idx, level, heading_text)."""
    headings: List[Tuple[int, int, str]] = []
    
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
            
        # Markdown-style headings
        if line.startswith('#'):
            level = len(line) - len(line.lstrip('#'))
            heading = line.lstrip('#').strip()
            if heading:
                headings.append((i, level, heading))
            continue
        
        # Numbered headings: "1.", "1.1", "A.", "a)"
        if re.match(r'^[0-9]+(\.[0-9]+)*\.?\s+[A-Z]', line):
            level = line.count('.') + 1
            heading = re.sub(r'^[0-9]+(\.[0-9]+)*\.?\s*', '', line)
            headings.append((i, level, heading))
            continue
        
        # Letter headings: "A.", "a)", "I.", "i)"
        if re.match(r'^[A-Za-z]+[\.\)]\s+[A-Z]', line):
            heading = re.sub(r'^[A-Za-z]+[\.\)]\s*', '', line)
            headings.append((i, 2, heading))
            continue
        
        # All caps lines (likely headings)
        if len(line) > 5 and line.isupper() and not re.search(r'[0-9]{3,}', line):
            headings.append((i, 1, line))
            continue
    
    return headings


def build_heading_path(headings: List[Tuple[int, int, str]], current_line: int) -> List[str]:
    """Build hierarchical heading path for current line position."""
    path: List[str] = []
    active_headings: List[Tuple[int, str]] = []  # (level, text)
    
    for line_idx, level, text in headings:
        if line_idx >= current_line:
            break
        
        # Remove headings at same or deeper level
        active_headings = [(l, t) for l, t in active_headings if l < level]
        active_headings.append((level, text))
    
    return [text for _, text in active_headings]


def chunk_text_by_headers(text: str, file_name: str, target_tokens: int = 200, overlap_ratio: float = 0.15) -> List[Chunk]:
    """Split text into chunks respecting heading boundaries."""
    text = normalize_text(text)
    lines = text.split('\n')
    headings = detect_headings(lines)
    
    chunks: List[Chunk] = []
    current_chunk = []
    current_tokens = 0
    chunk_id = 0
    
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        
        # Estimate tokens (rough: words * 1.3)
        line_tokens = int(len(line.split()) * 1.3)
        
        # Check if we need to split
        if current_tokens + line_tokens > target_tokens and current_chunk:
            # Create chunk
            chunk_text = '\n'.join(current_chunk).strip()
            if chunk_text:
                heading_path = build_heading_path(headings, i)
                chunks.append(Chunk(
                    id=f"{file_name}:chunk:{chunk_id}",
                    text=chunk_text,
                    heading_path=heading_path,
                    metadata={
                        "source_file": file_name,
                        "chunk_index": chunk_id,
                        "heading_path": " > ".join(heading_path) if heading_path else "",
                        "token_count": current_tokens
                    }
                ))
                chunk_id += 1
            
            # Start new chunk with overlap
            overlap_lines = int(len(current_chunk) * overlap_ratio)
            current_chunk = current_chunk[-overlap_lines:] if overlap_lines > 0 else []
            current_tokens = sum(int(len(l.split()) * 1.3) for l in current_chunk)
        
        current_chunk.append(line)
        current_tokens += line_tokens
    
    # Final chunk
    if current_chunk:
        chunk_text = '\n'.join(current_chunk).strip()
        if chunk_text:
            heading_path = build_heading_path(headings, len(lines))
            chunks.append(Chunk(
                id=f"{file_name}:chunk:{chunk_id}",
                text=chunk_text,
                heading_path=heading_path,
                metadata={
                    "source_file": file_name,
                    "chunk_index": chunk_id,
                    "heading_path": " > ".join(heading_path) if heading_path else "",
                    "token_count": current_tokens
                }
            ))
    
    return chunks
