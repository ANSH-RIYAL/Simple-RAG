import re
from dataclasses import dataclass
from typing import List, Literal


Intent = Literal["chitchat", "kb_qa", "list", "table"]


@dataclass
class QueryProcessingResult:
    original_query: str
    normalized_query: str
    intent: Intent
    should_search: bool
    rewritten_query: str
    keyword_boost_terms: List[str]


def normalize_query(q: str) -> str:
    q = q.replace("\u2019", "'")
    q = q.strip()
    q = re.sub(r"\s+", " ", q)
    return q


def detect_intent(q: str) -> Intent:
    lower = q.lower()
    # Chitchat / greetings
    if re.search(r"\b(hi|hello|hey|how are you|good morning|good evening)\b", lower):
        return "chitchat"
    # Lists / tables cues
    if re.search(r"\b(list|bullet|top \d+|enumerate|table of|compare|vs\.?|versus)\b", lower):
        return "list"
    if re.search(r"\btable\b", lower):
        return "table"
    return "kb_qa"


_DOMAIN_SYNONYMS = {
    "mfa": ["multi-factor authentication", "two-factor authentication"],
    "sso": ["single sign-on"],
    "rto": ["recovery time objective"],
    "rpo": ["recovery point objective"],
    "soc2": ["soc 2", "service organization control 2"],
    "encryption at rest": ["data at rest encryption", "aes-256"],
    "encryption in transit": ["tls", "https", "ssl"],
    "incident response": ["security incident handling", "breach notification"],
    "backup": ["disaster recovery", "restore testing"],
}


def extract_keywords(q: str) -> List[str]:
    # Preserve domain terms as-is, capture quoted phrases, and alphanumeric tokens of length>=3
    phrases = re.findall(r'"([^"]+)"|\'([^\']+)\'', q)
    captured: List[str] = []
    for ph in phrases:
        # regex returns tuples; pick non-empty
        text = ph[0] or ph[1]
        if text:
            captured.append(text)
    # Remove quotes from q for the rest
    q_wo_quotes = re.sub(r'"[^\"]+"|\'[^\']+\'', " ", q)
    tokens = re.findall(r"[A-Za-z0-9][A-Za-z0-9\-_/\.]{1,}", q_wo_quotes)
    tokens = [t for t in tokens if len(t) >= 3]
    # Deduplicate while preserving order
    seen = set()
    out: List[str] = []
    for t in captured + tokens:
        t_norm = t.strip()
        if t_norm.lower() not in seen:
            seen.add(t_norm.lower())
            out.append(t_norm)
    return out


def expand_synonyms(keywords: List[str]) -> List[str]:
    expanded: List[str] = []
    seen = set()
    for kw in keywords:
        lower = kw.lower()
        expanded.append(kw)
        if lower in _DOMAIN_SYNONYMS:
            for syn in _DOMAIN_SYNONYMS[lower]:
                if syn not in seen:
                    expanded.append(syn)
                    seen.add(syn)
    # Also map embedded domain abbreviations in tokens
    for key, syns in _DOMAIN_SYNONYMS.items():
        if any(key in k.lower() for k in keywords):
            for syn in syns:
                if syn not in expanded:
                    expanded.append(syn)
    return expanded


def rewrite_query(q: str, keywords: List[str]) -> str:
    # Build a retrieval-optimized query: original (normalized) + key phrases + synonyms
    normalized = normalize_query(q)
    syns = expand_synonyms([k for k in keywords if len(k) <= 40])
    # Keep unique order
    unique_syns: List[str] = []
    seen = set()
    for s in syns:
        s_norm = s.strip()
        if s_norm.lower() not in seen:
            unique_syns.append(s_norm)
            seen.add(s_norm.lower())
    # Join with separators that work well for basic tokenizers
    if unique_syns:
        return normalized + " | " + " | ".join(unique_syns)
    return normalized


def process_query(q: str) -> QueryProcessingResult:
    normalized = normalize_query(q)
    intent = detect_intent(normalized)
    should_search = intent != "chitchat"
    keywords = extract_keywords(normalized)
    rewritten = rewrite_query(normalized, keywords)
    return QueryProcessingResult(
        original_query=q,
        normalized_query=normalized,
        intent=intent,
        should_search=should_search,
        rewritten_query=rewritten,
        keyword_boost_terms=keywords,
    )

