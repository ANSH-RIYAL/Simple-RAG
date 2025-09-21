from dataclasses import dataclass
from typing import List
import re


@dataclass
class Chunk:
    id: str
    text: str
    heading_path: List[str]
    metadata: dict


def extract_header_from_chunk(chunk_text: str) -> str:
    """Extract the most prominent header from a chunk using simple regex patterns."""
    # Common header patterns
    patterns = [
        r'^#+\s+(.+)$',           # Markdown headers: ## Header
        r'^\d+\.\d+\s+(.+)$',     # Numbered sections: 1.1 Header
        r'^[A-Z][A-Z\s]{10,}$',   # ALL CAPS headers
        r'^[A-Z][a-z]+(\s+[A-Z][a-z]+)*:$',  # Title Case: Header:
    ]
    
    lines = chunk_text.split('\n')
    for line in lines[:5]:  # Only check first 5 lines
        line = line.strip()
        if len(line) < 10:  # Skip very short lines
            continue
            
        for pattern in patterns:
            match = re.match(pattern, line, re.MULTILINE)
            if match:
                header = match.group(1).strip() if match.groups() else line.strip()
                # Clean up the header
                header = re.sub(r'[^\w\s-]', '', header)  # Remove special chars
                return header[:50]  # Limit length
    
    return ""  # No header found


def simple_chunk_text(text: str, file_name: str, chunk_size: int = 1000, overlap: int = 100) -> List[Chunk]:
    """Simple text chunking - just split by character count with overlap."""
    text = text.strip()
    if not text:
        return []
    
    chunks = []
    start = 0
    chunk_index = 0
    
    while start < len(text):
        end = start + chunk_size
        
        # If we're not at the end, try to break at a sentence or paragraph
        if end < len(text):
            # Look for paragraph break first
            para_break = text.rfind('\n\n', start, end)
            if para_break != -1 and para_break > start + chunk_size // 2:
                end = para_break + 2
            else:
                # Look for sentence break
                sent_break = text.rfind('. ', start, end)
                if sent_break != -1 and sent_break > start + chunk_size // 2:
                    end = sent_break + 2
        
        chunk_text = text[start:end].strip()
        if chunk_text:
            # Extract header from this chunk
            header = extract_header_from_chunk(chunk_text)
            
            chunks.append(Chunk(
                id=f"{file_name}:chunk:{chunk_index}",
                text=chunk_text,
                heading_path=[header] if header else [],
                metadata={
                    "source_file": file_name,
                    "chunk_index": chunk_index,
                    "heading_path": header,
                    "char_start": start,
                    "char_end": end,
                    "char_count": len(chunk_text)
                }
            ))
            chunk_index += 1
        
        # Move start position with overlap
        start = max(start + chunk_size - overlap, end)
    
    return chunks


def chunk_text_by_headers(text: str, file_name: str, target_tokens: int = 50, overlap_ratio: float = 0.25) -> List[Chunk]:
    """Simple chunking with basic header detection from chunk content."""
    # Convert token target to character estimate (roughly 4 chars per token)
    chunk_size = target_tokens * 6
    overlap = int(chunk_size * overlap_ratio)
    
    return simple_chunk_text(text, file_name, chunk_size, overlap)