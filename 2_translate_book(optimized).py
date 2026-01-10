#!/usr/bin/env python3
"""
Book Translation Script (Optimized Version)
Simplified and optimized version with OpenRouter API only.
"""

import json
import re
import time
import requests
import os
from typing import Dict, List, Optional, Any
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ============================================================================
# API CONFIGURATION
# ============================================================================
API_URL = "https://openrouter.ai/api/v1/chat/completions"
API_KEY = os.getenv("OPENROUTER_API_KEY")
MODEL = os.getenv("OPENROUTER_MODEL", "google/gemini-3-flash-preview")

# Default values (will be overridden by config if available)
MAX_TOKENS_PER_BATCH = 5000
CONTEXT_PARAGRAPHS_COUNT = 3


def load_json(file_path: str) -> Dict[str, Any]:
    """Load JSON file with existence check."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(file_path: str, data: Dict[str, Any]) -> None:
    """Save JSON file."""
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_text_file(file_path: str) -> Optional[str]:
    """Load text file if it exists, return None otherwise."""
    path = Path(file_path)
    if not path.exists():
        return None
    try:
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            return content if content else None
    except Exception as e:
        print(f"   Warning: Could not load {file_path}: {e}")
        return None


def estimate_tokens(text: str) -> int:
    """Estimate token count (roughly 1 token ≈ 3-4 characters)."""
    return len(text) // 3


def split_text_into_batches(text: str, max_tokens: int) -> List[str]:
    """
    Split text into logical batches, preserving paragraph structure.
    Always splits at paragraph boundaries (\n\n) to ensure complete logical units.
    Returns list of batch texts.
    """
    if not text:
        return []
    
    if estimate_tokens(text) <= max_tokens:
        return [text]
    
    # Split into paragraphs first
    paragraphs = re.split(r'(\n\n+)', text)
    
    batches = []
    current_batch = []
    current_tokens = 0
    
    i = 0
    while i < len(paragraphs):
        para = paragraphs[i]
        para_tokens = estimate_tokens(para)
        
        # If single paragraph exceeds max_tokens, we need to split it
        if para_tokens > max_tokens and not current_batch:
            # Try to split at sentence boundaries
            sentences = re.split(r'([.!?]+\s+)', para)
            sentence_batch = []
            sentence_tokens = 0
            
            for j, sent in enumerate(sentences):
                sent_tokens = estimate_tokens(sent)
                if sentence_tokens + sent_tokens > max_tokens and sentence_batch:
                    batches.append(''.join(sentence_batch))
                    sentence_batch = [sent]
                    sentence_tokens = sent_tokens
                else:
                    sentence_batch.append(sent)
                    sentence_tokens += sent_tokens
            
            if sentence_batch:
                current_batch = sentence_batch
                current_tokens = sentence_tokens
            i += 1
            continue
        
        # Check if adding this paragraph would exceed limit
        if current_tokens + para_tokens > max_tokens and current_batch:
            # Save current batch and start new one
            batches.append(''.join(current_batch))
            current_batch = [para]
            current_tokens = para_tokens
        else:
            current_batch.append(para)
            current_tokens += para_tokens
        
        i += 1
    
    # Don't forget the last batch
    if current_batch:
        batches.append(''.join(current_batch))
    
    return batches


def format_glossary_for_prompt(glossary: Dict[str, Any]) -> str:
    """Format glossary for inclusion in translation prompt."""
    lines = ["ГЛОССАРИЙ:"]
    
    categories = {
        "characters": "Персонажи",
        "locations": "Места", 
        "items": "Предметы",
        "organizations": "Организации",
        "terms": "Термины"
    }
    
    for key, title in categories.items():
        if key in glossary and glossary[key]:
            lines.append(f"\n{title}:")
            for en, ru in glossary[key].items():
                lines.append(f"  {en} → {ru}")
    
    return "\n".join(lines)


def get_last_paragraphs(text: str, count: int = 3) -> str:
    """Extract last N paragraphs from text for context."""
    if not text:
        return ""
    
    paragraphs = [p.strip() for p in re.split(r'\n\n+', text) if p.strip()]
    if not paragraphs:
        return ""
    
    return '\n\n'.join(paragraphs[-count:])


def call_api(prompt: str, max_tokens: int = 60000, max_retries: int = 3, retry_timeout: int = 60) -> str:
    """Make API call to OpenRouter. Returns translated text."""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }
    
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
        "max_tokens": max_tokens
    }
    
    for attempt in range(1, max_retries + 1):
        try:
            response = requests.post(API_URL, headers=headers, json=payload, timeout=180)
            response.raise_for_status()
            
            result = response.json()
            if "choices" not in result or not result["choices"]:
                raise ValueError("Invalid API response: no choices")
            
            choice = result["choices"][0]
            text = choice["message"]["content"]
            
            if choice.get("finish_reason") == "length":
                print(f"    Warning: Response truncated")
            
            # Clean up response
            text = text.strip() if text else ""
            if text.startswith("```"):
                text = re.sub(r'^```[a-z]*\n', '', text)
                text = re.sub(r'\n```$', '', text)
                text = text.strip()
            
            if text:
                return text
            
            if attempt < max_retries:
                print(f"    Empty response (attempt {attempt}/{max_retries}). Retrying in {retry_timeout}s...")
                time.sleep(retry_timeout)
            else:
                raise Exception(f"Empty response after {max_retries} attempts")
                
        except requests.exceptions.RequestException as e:
            if attempt < max_retries:
                print(f"    API error (attempt {attempt}/{max_retries}): {e}. Retrying...")
                time.sleep(retry_timeout)
            else:
                raise Exception(f"API failed after {max_retries} attempts: {e}")
        except Exception as e:
            if attempt < max_retries:
                print(f"    Error (attempt {attempt}/{max_retries}): {e}. Retrying...")
                time.sleep(retry_timeout)
            else:
                raise
    
    raise Exception("Translation failed")


def translate_text(text: str, glossary: Dict[str, Any], style_guide: Optional[str] = None, 
                   context: Optional[str] = None, preserve_paragraphs: bool = True) -> str:
    """
    Translate text using OpenRouter API.
    Universal function for both short fields and long content.
    """
    if not text:
        return text
    
    glossary_text = format_glossary_for_prompt(glossary)
    
    # Build optional sections
    style_section = f"\nСТИЛИСТИЧЕСКИЙ ГАЙД:\n{style_guide}\n" if style_guide else ""
    context_section = f"\nКОНТЕКСТ:\n{context}\n" if context else ""
    
    # Paragraph requirements only for multi-paragraph text
    para_count = len(re.findall(r'\n\n+', text))
    para_section = ""
    if preserve_paragraphs and para_count > 0:
        para_section = f"""
ВАЖНО - СТРУКТУРА АБЗАЦЕВ:
- В оригинале {para_count} абзацев (разделены двойным переносом строки \\n\\n)
- В переводе ДОЛЖНО БЫТЬ РОВНО {para_count} абзацев
- НЕ объединяй абзацы
- НЕ разбивай абзацы на несколько
- Каждый абзац оригинала = один абзац перевода
"""
    
    prompt = f"""Ты профессиональный переводчик художественной литературы с английского на русский.

{glossary_text}{style_section}{context_section}{para_section}
ИНСТРУКЦИИ:
1. Переведи текст на русский язык
2. Сохрани художественный стиль оригинала
3. Используй глоссарий для имён и терминов
4. Верни ТОЛЬКО перевод, без комментариев

Текст для перевода:

{text}"""

    print(f"    Prompt: ~{estimate_tokens(prompt)} tokens")
    
    translated = call_api(prompt)
    
    # Validate paragraph count for multi-paragraph text
    if preserve_paragraphs and para_count > 0:
        translated_count = len(re.findall(r'\n\n+', translated))
        if translated_count != para_count:
            diff = para_count - translated_count
            print(f"    Warning: Paragraph mismatch ({'+' if diff < 0 else ''}{-diff})")
    
    return translated


def translate_content(content: str, glossary: Dict[str, Any], style_guide: Optional[str] = None,
                     max_tokens_per_batch: int = 5000, context_paragraphs: int = 3) -> str:
    """Translate long content, splitting into batches if necessary."""
    if not content:
        return ""
    
    batches = split_text_into_batches(content, max_tokens_per_batch)
    
    if len(batches) == 1:
        print(f"    Single batch ({estimate_tokens(content)} tokens)...")
        return translate_text(content, glossary, style_guide=style_guide)
    
    # Multiple batches
    translated_parts = []
    prev_context = None
    
    for i, batch in enumerate(batches, 1):
        print(f"    Batch {i}/{len(batches)} ({estimate_tokens(batch)} tokens)...")
        
        context = get_last_paragraphs(prev_context, context_paragraphs) if prev_context else None
        
        translated = translate_text(batch, glossary, style_guide=style_guide, context=context)
        translated_parts.append(translated)
        prev_context = translated
        
        if i < len(batches):
            time.sleep(3)  # Rate limiting
    
    combined = "".join(translated_parts)
    
    # Final paragraph check
    orig_paras = len(re.findall(r'\n\n+', content))
    trans_paras = len(re.findall(r'\n\n+', combined))
    if orig_paras == trans_paras:
        print(f"    ✓ Structure preserved ({orig_paras} paragraphs)")
    
    return combined


def translate_field(text: str, glossary: Dict[str, Any], field_name: str, 
                   style_guide: Optional[str] = None) -> str:
    """Translate a short text field (title, author, etc.)."""
    if not text:
        return text
    
    print(f"  Translating {field_name}...")
    
    translated = translate_text(text, glossary, style_guide=style_guide, preserve_paragraphs=False)
    
    # Safety: ensure single-line fields stay single-line
    if '\n' not in text and '\n' in translated:
        translated = translated.split('\n')[0].strip()
    
    return translated


class TranslationState:
    """Manages translation state and checkpoints."""
    
    def __init__(self, checkpoint_path: str):
        self.checkpoint_path = checkpoint_path
        self.data = None
    
    def init_from_book(self, book_data: Dict[str, Any]) -> None:
        """Initialize translation data structure from book."""
        self.data = {
            "book_id": book_data["book_id"],
            "metadata": {
                "title": "",
                "author": "",
                "language_source": book_data["metadata"]["language_source"],
                "language_target": "ru"
            },
            "parts": []
        }
    
    def load_checkpoint(self) -> bool:
        """Load checkpoint if exists. Returns True if loaded."""
        if Path(self.checkpoint_path).exists():
            self.data = load_json(self.checkpoint_path)
            return True
        return False
    
    def save(self) -> None:
        """Save current state to checkpoint."""
        save_json(self.checkpoint_path, self.data)
    
    def cleanup(self) -> None:
        """Remove checkpoint file."""
        path = Path(self.checkpoint_path)
        if path.exists():
            os.remove(path)


def main():
    """Main translation function."""
    from config_helper import get_paths
    
    # Validate API key
    if not API_KEY:
        raise ValueError("OPENROUTER_API_KEY not found. Set it in .env file.")
    
    print("=" * 60)
    print("Book Translation Script (Optimized)")
    print(f"Model: {MODEL}")
    print("=" * 60)
    
    # Load configuration
    try:
        paths = get_paths()
        paths.ensure_work_dir()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please check your config.json file")
        exit(1)
    
    max_tokens = paths.translation_max_tokens
    context_paras = paths.translation_context
    
    print(f"Book: {paths.book}")
    print(f"Settings: max_tokens={max_tokens}, context_paragraphs={context_paras}")
    
    # Load data
    print("\n1. Loading files...")
    book_data = load_json(str(paths.output_json))
    glossary = load_json(str(paths.glossary_json))
    style_guide = load_text_file(str(paths.styleguide_md))
    
    print(f"   Book: {len(book_data.get('parts', []))} parts")
    print(f"   Glossary: {sum(len(v) for v in glossary.values() if isinstance(v, dict))} entries")
    print(f"   Style guide: {'loaded' if style_guide else 'not found'}")
    
    # Initialize state
    state = TranslationState(str(paths.checkpoint_json))
    if state.load_checkpoint():
        print("   Resuming from checkpoint...")
    else:
        state.init_from_book(book_data)
    
    # Translate metadata
    if not state.data["metadata"]["title"]:
        print("\n2. Translating metadata...")
        state.data["metadata"]["title"] = translate_field(
            book_data["metadata"]["title"], glossary, "title", style_guide
        )
        state.data["metadata"]["author"] = translate_field(
            book_data["metadata"]["author"], glossary, "author", style_guide
        )
        state.save()
    
    # Translate parts
    print("\n3. Translating parts...")
    
    for part_idx, part in enumerate(book_data.get("parts", []), 1):
        part_num = part.get("part_number", part_idx)
        chapters = part.get("chapters", [])
        
        # Check if part already fully translated
        if part_idx <= len(state.data.get("parts", [])):
            existing = state.data["parts"][part_idx - 1]
            if existing.get("chapters") and len(existing["chapters"]) == len(chapters):
                print(f"\n   Part {part_num} done, skipping...")
                continue
        
        print(f"\n   Part {part_num}: {part.get('part_title', 'Untitled')}")
        
        # Initialize or get part data
        if part_idx > len(state.data["parts"]):
            part_data = {"part_number": part_num, "part_title": "", "chapters": []}
            if "blockquote" in part:
                part_data["blockquote"] = {"text": "", "author": ""}
            state.data["parts"].append(part_data)
        else:
            part_data = state.data["parts"][part_idx - 1]
        
        # Translate part title
        if not part_data.get("part_title"):
            part_data["part_title"] = translate_field(part["part_title"], glossary, "part_title", style_guide)
            state.save()
        
        # Translate blockquote
        if "blockquote" in part and part["blockquote"] and not part_data.get("blockquote", {}).get("text"):
            print(f"  Translating blockquote...")
            part_data["blockquote"] = {
                "text": translate_field(part["blockquote"]["text"], glossary, "blockquote", style_guide),
                "author": translate_field(part["blockquote"]["author"], glossary, "author", style_guide)
            }
            state.save()
        
        # Translate chapters
        for ch_idx, chapter in enumerate(chapters, 1):
            ch_id = chapter.get("chapter_id", f"ch{ch_idx}")
            
            # Check if already translated
            if ch_idx <= len(part_data.get("chapters", [])):
                if part_data["chapters"][ch_idx - 1].get("content"):
                    print(f"    Chapter {ch_idx} done, skipping...")
                    continue
            
            print(f"    Chapter {ch_idx}/{len(chapters)} ({ch_id}): {chapter.get('title', '')}")
            
            # Initialize chapter data
            if ch_idx > len(part_data["chapters"]):
                ch_data = {"chapter_id": ch_id, "title": "", "content": ""}
                part_data["chapters"].append(ch_data)
            else:
                ch_data = part_data["chapters"][ch_idx - 1]
            
            # Translate title
            if not ch_data.get("title"):
                ch_data["title"] = translate_field(chapter["title"], glossary, "chapter title", style_guide)
                state.save()
            
            # Translate content
            if not ch_data.get("content"):
                print(f"      Content ({chapter.get('word_count', 0)} words)...")
                ch_data["content"] = translate_content(
                    chapter["content"], glossary, style_guide,
                    max_tokens_per_batch=max_tokens, context_paragraphs=context_paras
                )
                state.save()
                print(f"      Saved.")
        
        print(f"   Part {part_num} completed.")
    
    # Save final result
    print("\n4. Saving final translation...")
    save_json(str(paths.translated_json), state.data)
    print(f"   Saved to: {paths.translated_json}")
    
    # Cleanup
    print("\n5. Cleanup...")
    state.cleanup()
    print("   Checkpoint removed.")
    
    print("\n" + "=" * 60)
    print("Translation completed!")
    print("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted. Progress saved.")
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()
        print("\nProgress saved. Run again to resume.")