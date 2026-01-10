#!/usr/bin/env python3
"""
EPUB to JSON Parser (Optimized)
Parses an EPUB book and converts it to JSON format matching JSON Map.json structure.
Supports both .epub files and extracted EPUB directories.
"""

import json
import re
import zipfile
import tempfile
import shutil
import xml.etree.ElementTree as ET
from itertools import count
from pathlib import Path
from html.parser import HTMLParser
from typing import Dict, List, Optional, Any, Tuple


class TextExtractor(HTMLParser):
    """HTML parser to extract text content from XHTML files."""
    
    def __init__(self):
        super().__init__()
        self.text_parts = []
        self.current_tag = None
        self.in_transition = False
        self.transition_inserted = False
    
    def _insert_transition_marker(self):
        """Insert transition marker (* * *) with proper spacing."""
        if self.transition_inserted:
            return
        # Ensure proper spacing before marker
        if self.text_parts and not self.text_parts[-1].endswith('\n\n'):
            if not self.text_parts[-1].endswith('\n'):
                self.text_parts.append('\n')
            self.text_parts.append('\n')
        self.text_parts.append('* * *')
        self.text_parts.append('\n\n')
        self.transition_inserted = True
        
    def handle_data(self, data):
        """Collect text data."""
        if self.in_transition:
            return
        if self.current_tag not in ['script', 'style']:
            self.text_parts.append(data)
    
    def handle_starttag(self, tag, attrs):
        """Track current tag."""
        attrs_dict = dict(attrs)
        is_transition_div = (tag == 'div' and attrs_dict.get('class') == 'transition')
        is_transition_hr = (tag == 'hr' and attrs_dict.get('class') == 'transition')
        
        if is_transition_div or is_transition_hr:
            self._insert_transition_marker()
            if is_transition_div:
                self.in_transition = True
            return
        
        if tag == 'p' and self.transition_inserted:
            self.transition_inserted = False
        
        self.current_tag = tag
        if tag in ['p', 'div', 'br', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
            if self.text_parts and not self.text_parts[-1].endswith('\n'):
                self.text_parts.append('\n')
    
    def handle_startendtag(self, tag, attrs):
        """Handle self-closing tags."""
        if tag == 'hr' and dict(attrs).get('class') == 'transition':
            self._insert_transition_marker()
    
    def handle_endtag(self, tag):
        """Handle end tags."""
        if self.in_transition:
            if tag == 'div':
                self.in_transition = False
            return
            
        if tag in ['p', 'div']:
            if self.text_parts and not self.text_parts[-1].endswith('\n'):
                self.text_parts.append('\n')
            if tag == 'p':
                self.transition_inserted = False
        self.current_tag = None
    
    def get_text(self) -> str:
        """Return extracted text."""
        text = ''.join(self.text_parts)
        return normalize_whitespace(text)


def find_epub_file(epub_dir: str, extension: str, default_name: str) -> Path:
    """
    Find file with given extension in EPUB directory.
    
    Args:
        epub_dir: Base directory of EPUB files
        extension: File extension to search for (e.g., '.opf', '.ncx')
        default_name: Default filename (e.g., 'content.opf', 'toc.ncx')
        
    Returns:
        Path to found file
        
    Raises:
        FileNotFoundError if file not found
    """
    epub_path = Path(epub_dir)
    search_dirs = [epub_path, epub_path / 'OEBPS']
    
    # Check default locations first
    for search_dir in search_dirs:
        default_path = search_dir / default_name
        if default_path.exists():
            return default_path
    
    # Search for any file with the extension
    for search_dir in search_dirs:
        if search_dir.exists():
            for file in search_dir.iterdir():
                if file.suffix == extension:
                    return file
    
    raise FileNotFoundError(f"{default_name} not found in {epub_dir}")


def resolve_content_path(epub_dir: str, content_src: str) -> Path:
    """
    Resolve content source path to actual file path.
    
    Args:
        epub_dir: Base directory of EPUB files
        content_src: Content source from NCX/OPF file
        
    Returns:
        Resolved Path to the content file
    """
    epub_path = Path(epub_dir)
    
    # Remove fragment identifier if present (e.g., "file.xhtml#section1")
    content_src = content_src.split('#')[0]
    
    # Handle different EPUB path structures
    if content_src.startswith('OEBPS/'):
        return epub_path / content_src
    elif content_src.startswith('Text/'):
        return epub_path / 'OEBPS' / content_src
    elif content_src.startswith('xhtml/'):
        return epub_path / 'OEBPS' / content_src
    elif content_src.startswith('text/'):
        # Common in HarperCollins EPUBs - text/ folder inside OEBPS
        return epub_path / 'OEBPS' / content_src
    elif '/' in content_src:
        # Try direct path first
        direct_path = epub_path / content_src
        if direct_path.exists():
            return direct_path
        # Try inside OEBPS
        oebps_path = epub_path / 'OEBPS' / content_src
        if oebps_path.exists():
            return oebps_path
        return direct_path
    
    # Just filename - try common locations
    possible_paths = [
        epub_path / 'Text' / content_src,
        epub_path / 'OEBPS' / 'xhtml' / content_src,
        epub_path / 'OEBPS' / 'Text' / content_src,
        epub_path / 'OEBPS' / 'text' / content_src,
    ]
    
    for path in possible_paths:
        if path.exists():
            return path
    
    # Search in all subdirectories
    for path in epub_path.rglob(content_src):
        return path
    
    # Return first possible path as fallback
    return possible_paths[0]


def get_text_or_default(elem, default: str = '') -> str:
    """Extract text from XML element or return default."""
    return elem.text if elem is not None and elem.text else default


def parse_metadata(opf_path: Path) -> Dict[str, Any]:
    """
    Parse metadata from content.opf file.
    
    Args:
        opf_path: Path to content.opf file
        
    Returns:
        Dictionary with metadata (title, author, language_source, language_target)
    """
    tree = ET.parse(opf_path)
    root = tree.getroot()
    
    ns = {
        'dc': 'http://purl.org/dc/elements/1.1/',
        'opf': 'http://www.idpf.org/2007/opf'
    }
    
    # Extract metadata with defaults
    title = get_text_or_default(root.find('.//dc:title', ns))
    
    creator_elem = root.find('.//dc:creator[@opf:role="aut"]', ns)
    if creator_elem is None:
        creator_elem = root.find('.//dc:creator', ns)
    author = get_text_or_default(creator_elem)
    
    language = get_text_or_default(root.find('.//dc:language', ns), 'en')
    
    return {
        'title': title,
        'author': author,
        'language_source': language,
        'language_target': language
    }


def roman_to_int(roman: str) -> int:
    """Convert Roman numeral to integer."""
    roman_map = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}
    result = 0
    prev_value = 0
    
    for char in reversed(roman.upper()):
        value = roman_map.get(char, 0)
        if value < prev_value:
            result -= value
        else:
            result += value
        prev_value = value
    
    return result


def word_to_number(word: str) -> Optional[int]:
    """Convert word number (One, Two, Three, etc.) to integer."""
    word_map = {
        'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
        'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10,
        'eleven': 11, 'twelve': 12, 'thirteen': 13, 'fourteen': 14, 'fifteen': 15,
        'sixteen': 16, 'seventeen': 17, 'eighteen': 18, 'nineteen': 19, 'twenty': 20
    }
    return word_map.get(word.lower())


# Constants
CHAPTER_TITLE_SEARCH_LENGTH = 500  # Number of characters to search for chapter titles

# Metadata keywords to exclude from content chapters
METADATA_KEYWORDS = frozenset([
    'cover', 'title page', 'copyright', 'contents', 'dedication',
    'acknowledgments', 'acknowledgements', 'about the author', 'other titles', 'epigraph',
    'maps', 'preface', 'author’s note', 'by robert jackson bennett', 'military ranks',
    'also by', 'praise for', 'newsletter', 'navigation'
])

# Content keywords that indicate actual book content
CONTENT_KEYWORDS = frozenset(['chapter', 'epilogue', 'prologue', 'interlude', 'part'])


def is_content_chapter(label_text: str) -> bool:
    """Determine if a navPoint represents a content chapter."""
    label_lower = label_text.lower()
    
    # If it explicitly matches a content keyword, it's content
    if any(kw in label_lower for kw in CONTENT_KEYWORDS):
        return True
        
    # If it matches metadata keyword, it's not content
    if any(kw in label_lower for kw in METADATA_KEYWORDS):
        return False
    
    # Default to True if we're not sure (better to include extra than miss content)
    # But maybe we can be stricter if needed
    return True


def determine_chapter_type_and_id(label_text: str) -> Tuple[Optional[str], Optional[int], Optional[str]]:
    """
    Determine chapter type and generate chapter_id from label text.
    
    Returns:
        Tuple of (chapter_type, chapter_num, chapter_id)
    """
    label_lower = label_text.lower()
    
    # Check for special chapter types
    if 'epilogue' in label_lower:
        return ('epilogue', None, 'epilogue')
    
    if 'prologue' in label_lower:
        return ('prologue', None, 'prologue')
    
    if 'interlude' in label_lower:
        match = re.search(r'interlude\s+(\d+)', label_lower)
        if match:
            num = int(match.group(1))
            return ('interlude', num, f'interlude_{num}')
        return ('interlude', None, 'interlude')
    
    if 'chapter' in label_lower:
        # Try numeric: "Chapter 1"
        match = re.search(r'chapter\s+(\d+)', label_lower)
        if match:
            num = int(match.group(1))
            return ('chapter', num, f'ch{num}')
        
        # Try Roman numeral: "Chapter I"
        match = re.search(r'chapter\s+([ivx]+)', label_lower)
        if match:
            num = roman_to_int(match.group(1))
            return ('chapter', num, f'ch{num}_intro')
        
        # Try word number: "Chapter One"
        match = re.search(r'chapter\s+([a-z]+)', label_lower)
        if match:
            num = word_to_number(match.group(1))
            if num is not None:
                return ('chapter', num, f'ch{num}')
    
    return (None, None, None)


def parse_toc(ncx_path: Path, epub_dir: str) -> List[Dict[str, Any]]:
    """
    Parse table of contents from toc.ncx file.
    
    Args:
        ncx_path: Path to toc.ncx file
        epub_dir: Base directory of EPUB files
        
    Returns:
        List of parts with their chapters
    """
    tree = ET.parse(ncx_path)
    root = tree.getroot()
    ns = {'ncx': 'http://www.daisy.org/z3986/2005/ncx/'}
    
    parts = []
    current_part = None
    has_parts = False
    
    def process_nav_point(nav_point, parent_is_part=False):
        nonlocal current_part, has_parts
        
        nav_label = nav_point.find('ncx:navLabel/ncx:text', ns)
        content = nav_point.find('ncx:content', ns)
        
        if nav_label is None or content is None:
            return
        
        label_text = nav_label.text or ''
        content_src = content.get('src', '')
        nav_point_class = nav_point.get('class', '')
        is_chapter_class = (nav_point_class == 'chapter')
        
        # Check if this is a part
        if 'Part' in label_text and ':' in label_text:
            has_parts = True
            if current_part is not None:
                parts.append(current_part)
            
            part_match = re.search(r'Part\s+([IVX]+):\s*(.+)', label_text)
            if part_match:
                part_roman = part_match.group(1)
                part_title = part_match.group(2).strip()
                part_number = roman_to_int(part_roman)
                
                current_part = {
                    'part_number': part_number,
                    'part_title': f"Part {part_roman}: {part_title}",
                    'part_file': str(resolve_content_path(epub_dir, content_src)),
                    'chapters': []
                }
            
            # Process child navPoints
            for child in nav_point.findall('ncx:navPoint', ns):
                process_nav_point(child, parent_is_part=True)
        
        elif parent_is_part or is_content_chapter(label_text) or is_chapter_class:
            # Skip if it's definitely metadata (regardless of class)
            if any(kw in label_text.lower() for kw in METADATA_KEYWORDS):
                print(f"Skipping metadata chapter: {label_text}")
                return

            # Create virtual part if book has no explicit parts
            if not has_parts and current_part is None:
                current_part = {
                    'part_number': 1,
                    'part_title': 'Chapters',
                    'part_file': None,
                    'chapters': []
                }
            
            if current_part is not None:
                chapter_type, chapter_num, chapter_id = determine_chapter_type_and_id(label_text)
                
                # Generate generic ID if needed
                if chapter_id is None:
                    label_slug = re.sub(r'[^a-zA-Z0-9]', '_', label_text.lower()).strip('_')
                    chapter_id = f"content_{label_slug}" if label_slug else f"content_{len(current_part['chapters']) + 1}"
                
                current_part['chapters'].append({
                    'chapter_id': chapter_id,
                    'title': label_text.strip(),
                    'file': str(resolve_content_path(epub_dir, content_src))
                })
            
            # Process child navPoints (chapters) of this chapter
            for child in nav_point.findall('ncx:navPoint', ns):
                process_nav_point(child, parent_is_part=parent_is_part)
    
    nav_map = root.find('ncx:navMap', ns)
    if nav_map is not None:
        for nav_point in nav_map.findall('ncx:navPoint', ns):
            process_nav_point(nav_point)
    
    if current_part is not None:
        parts.append(current_part)
    
    return parts


def extract_blockquote(xhtml_path: str) -> Optional[Dict[str, str]]:
    """Extract blockquote (epigraph) from a part XHTML file."""
    try:
        content = Path(xhtml_path).read_text(encoding='utf-8')
        
        blockquote_match = re.search(r'<blockquote[^>]*>(.*?)</blockquote>', content, re.DOTALL | re.IGNORECASE)
        if not blockquote_match:
            return None
        
        parser = TextExtractor()
        parser.feed(blockquote_match.group(1))
        blockquote_text = parser.get_text()
        
        # Try to extract author
        author_match = re.search(r'—([^—]+?)(?:,\s*[^—]+)?$', blockquote_text, re.MULTILINE)
        if author_match:
            author = author_match.group(1).strip()
            text = blockquote_text[:author_match.start()].strip()
        else:
            author_match = re.search(r'—\s*([A-Z\s,\.]+(?:,\s*[A-Z\s,\.]+)*)', blockquote_text)
            if author_match:
                author = author_match.group(1).strip()
                text = blockquote_text[:author_match.start()].strip()
            else:
                author = ""
                text = blockquote_text
        
        text = re.sub(r'\s+', ' ', text).strip()
        author = re.sub(r'\s+', ' ', author).strip()
        
        return {'text': text, 'author': author} if text else None
    except Exception as e:
        print(f"Error extracting blockquote from {xhtml_path}: {e}")
        return None


# Pre-compiled regex patterns for whitespace normalization
RE_SPACES_TABS = re.compile(r'[ \t]+')
RE_MULTIPLE_NEWLINES = re.compile(r'\n{3,}')
RE_TRAILING_SPACES = re.compile(r' +\n')
RE_LEADING_SPACES = re.compile(r'\n +')

# Pre-compiled chapter title patterns for removal
CHAPTER_TITLE_PATTERNS = [
    re.compile(r'^Chapter\s+\d+(?:\s*,\s*[^\n]+)?\s*\n+', re.IGNORECASE | re.MULTILINE),
    re.compile(r'^Chapter\s+[IVX]+(?:\s*,\s*[^\n]+)?\s*\n+', re.IGNORECASE | re.MULTILINE),
    re.compile(r'^(?:Chapter|CHAPTER)\s+[^\n]+\s*\n+', re.MULTILINE),
]

# Word numbers for chapter titles
WORD_NUMBERS = [
    'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten',
    'eleven', 'twelve', 'thirteen', 'fourteen', 'fifteen', 'sixteen', 'seventeen',
    'eighteen', 'nineteen', 'twenty'
]

# Pre-compiled word number patterns
WORD_NUMBER_PATTERNS = [
    re.compile(rf'^Chapter\s+{word}(?:\s*,\s*[^\n]+)?\s*\n+', re.IGNORECASE | re.MULTILINE)
    for word in WORD_NUMBERS
]


def normalize_whitespace(text: str) -> str:
    """Normalize whitespace in text while preserving paragraph breaks."""
    text = RE_SPACES_TABS.sub(' ', text)
    text = RE_MULTIPLE_NEWLINES.sub('\n\n', text)
    text = RE_TRAILING_SPACES.sub('\n', text)
    text = RE_LEADING_SPACES.sub('\n', text)
    return text.strip()


def extract_chapter_content(xhtml_path: str) -> str:
    """Extract text content from a chapter XHTML file."""
    try:
        content = Path(xhtml_path).read_text(encoding='utf-8')
        
        # Protect angle brackets in dialogue
        angle_bracket_placeholders = {}
        placeholder_counter = count()
        
        def protect_angle_brackets(match):
            placeholder = f"__ANGLE_DIALOGUE_{next(placeholder_counter)}__"
            angle_bracket_placeholders[placeholder] = match.group(0)
            return placeholder
        
        content = re.sub(r'&lt;([^&<>]+?)&gt;', protect_angle_brackets, content)
        
        parser = TextExtractor()
        parser.feed(content)
        text = parser.get_text()
        
        # Restore protected angle brackets
        for placeholder, original in angle_bracket_placeholders.items():
            restored = original.replace('&lt;', '<').replace('&gt;', '>')
            text = text.replace(placeholder, restored)
        
        # Remove page break markers
        text = re.sub(r'page_\d+', '', text)
        
        # Remove standalone chapter numbers
        text = re.sub(r'^\d+\s*$', '', text, flags=re.MULTILINE)
        
        # Remove chapter title patterns (apply to first N chars for efficiency)
        prefix_len = min(CHAPTER_TITLE_SEARCH_LENGTH, len(text))
        prefix = text[:prefix_len]
        suffix = text[prefix_len:]
        
        for pattern in CHAPTER_TITLE_PATTERNS:
            prefix = pattern.sub('', prefix)
        
        # Remove word number chapter titles
        for pattern in WORD_NUMBER_PATTERNS:
            prefix = pattern.sub('', prefix)
        
        text = prefix + suffix

        # Remove title/author repetition at start (common in some EPUBs)
        # Look for lines that are just the book title or "By Author"
        lines = text.split('\n')
        if lines:
            # Remove first line if it looks like a header (short, matches title/author patterns)
            first_line = lines[0].strip()
            if len(first_line) < 100 and (
                'by ' in first_line.lower() or
                ',' in first_line  # Often "Title, Book Name"
            ):
                lines = lines[1:]
                text = '\n'.join(lines)
        
        # Final cleanup using pre-compiled patterns
        text = RE_MULTIPLE_NEWLINES.sub('\n\n', text)
        text = RE_TRAILING_SPACES.sub('\n', text)
        text = RE_LEADING_SPACES.sub('\n', text)
        
        return text.strip()
    except Exception as e:
        print(f"Error extracting content from {xhtml_path}: {e}")
        return ""


def count_words(text: str) -> int:
    """Count words in text."""
    return len(text.split()) if text else 0


def load_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """Load configuration from config.json file."""
    if config_path is None:
        config_path = Path(__file__).parent / "config.json"
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    return json.loads(config_path.read_text(encoding='utf-8'))


def extract_epub(epub_file: str, extract_dir: Optional[str] = None) -> str:
    """
    Extract EPUB file to a directory.
    
    Args:
        epub_file: Path to .epub file
        extract_dir: Optional directory to extract to. If None, uses temp directory.
        
    Returns:
        Path to extracted directory
    """
    epub_path = Path(epub_file)
    
    if not epub_path.exists():
        raise FileNotFoundError(f"EPUB file not found: {epub_file}")
    
    if not zipfile.is_zipfile(epub_path):
        raise ValueError(f"Not a valid EPUB/ZIP file: {epub_file}")
    
    # Determine extraction directory
    if extract_dir:
        target_dir = Path(extract_dir)
    else:
        # Create temp directory with book name
        target_dir = Path(tempfile.mkdtemp(prefix=f"epub_{epub_path.stem}_"))
    
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract EPUB
    with zipfile.ZipFile(epub_path, 'r') as zf:
        zf.extractall(target_dir)
    
    print(f"   Extracted EPUB to: {target_dir}")
    return str(target_dir)


def prepare_epub_source(epub_source: str) -> Tuple[str, bool]:
    """
    Prepare EPUB source - extract if it's a file, or use directory as-is.
    
    Args:
        epub_source: Path to .epub file or extracted EPUB directory
        
    Returns:
        Tuple of (epub_directory_path, is_temp_dir)
        is_temp_dir indicates if the directory should be cleaned up after use
    """
    source_path = Path(epub_source)
    
    if source_path.is_file() and source_path.suffix.lower() == '.epub':
        # It's an EPUB file - extract it
        print(f"   Extracting EPUB file: {epub_source}")
        extracted_dir = extract_epub(epub_source)
        return extracted_dir, True
    elif source_path.is_dir():
        # It's already a directory
        return str(source_path), False
    else:
        raise ValueError(f"Invalid EPUB source: {epub_source}. Must be .epub file or directory.")


def parse_epub_to_json(epub_source: str, output_path: str, book_id: Optional[str] = None):
    """
    Main function to parse EPUB and convert to JSON.
    
    Args:
        epub_source: Path to .epub file OR directory containing extracted EPUB files
        output_path: Path to output JSON file
        book_id: Optional book ID (will generate if not provided)
    """
    # Prepare EPUB source (extract if needed)
    epub_dir, is_temp = prepare_epub_source(epub_source)
    epub_path = Path(epub_dir)
    
    try:
        # Parse metadata
        opf_path = find_epub_file(str(epub_path), '.opf', 'content.opf')
        metadata = parse_metadata(opf_path)
        
        # Generate book_id if not provided
        if book_id is None:
            title_slug = re.sub(r'[^a-zA-Z0-9]', '_', metadata['title']).lower()
            book_id = f"{title_slug}_{hash(metadata['author'])}"
        
        # Parse table of contents
        ncx_path = find_epub_file(str(epub_path), '.ncx', 'toc.ncx')
        parts_data = parse_toc(ncx_path, str(epub_path))
        
        # Build JSON structure
        result = {
            'book_id': book_id,
            'metadata': metadata,
            'parts': []
        }
        
        # Process each part
        for part_info in parts_data:
            part_data = {
                'part_number': part_info['part_number'],
                'part_title': part_info['part_title'],
                'chapters': []
            }
            
            # Extract blockquote if part file exists
            if part_info.get('part_file') and Path(part_info['part_file']).exists():
                blockquote = extract_blockquote(part_info['part_file'])
                if blockquote:
                    part_data['blockquote'] = blockquote
            
            # Process chapters
            for chapter_info in part_info['chapters']:
                chapter_path = Path(chapter_info['file'])
                if chapter_path.exists():
                    content = extract_chapter_content(str(chapter_path))
                    part_data['chapters'].append({
                        'chapter_id': chapter_info['chapter_id'],
                        'title': chapter_info['title'],
                        'content': content,
                        'word_count': count_words(content)
                    })
            
            result['parts'].append(part_data)
        
        # Simplify structure if there's only one virtual "Chapters" part
        if len(result['parts']) == 1 and result['parts'][0]['part_title'] == 'Chapters':
            # Remove the virtual part wrapper and move chapters to top level
            result['chapters'] = result['parts'][0]['chapters']
            total_chapters = len(result['chapters'])
            del result['parts']
            
            # Write JSON output
            Path(output_path).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
            print(f"Successfully parsed EPUB and created JSON at {output_path}")
            print(f"Found {total_chapters} chapters (no parts)")
        else:
            # Write JSON output
            Path(output_path).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
            
            total_chapters = sum(len(p['chapters']) for p in result['parts'])
            print(f"Successfully parsed EPUB and created JSON at {output_path}")
            print(f"Found {len(result['parts'])} parts with {total_chapters} chapters")
    
    finally:
        # Cleanup temp directory if we extracted the EPUB
        if is_temp and Path(epub_dir).exists():
            shutil.rmtree(epub_dir)
            print(f"   Cleaned up temp directory: {epub_dir}")


if __name__ == '__main__':
    from config_helper import get_paths
    
    try:
        paths = get_paths()
        paths.ensure_work_dir()
        
        print(f"Processing: {paths.book}")
        print(f"  EPUB: {paths.epub_source}")
        print(f"  Output: {paths.output_json}")
        
        parse_epub_to_json(
            str(paths.epub_source),
            str(paths.output_json),
            book_id=paths.book
        )
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please check your config.json file")
        exit(1)
