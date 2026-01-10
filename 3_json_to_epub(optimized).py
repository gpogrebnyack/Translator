#!/usr/bin/env python3
"""
JSON to EPUB Converter (Optimized)
Converts JSON to EPUB format with proper formatting, cover, and epigraphs.
Supports both structured (with parts) and flat (chapters only) JSON formats.
"""

import json
import os
import html
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from ebooklib import epub

# Constants
MIME_TYPES = {
    '.jpg': 'image/jpeg',
    '.jpeg': 'image/jpeg',
    '.png': 'image/png'
}

CSS_CONTENT = """
@namespace epub "http://www.idpf.org/2007/ops";

body {
    font-family: Georgia, serif;
    line-height: 1.6;
    margin: 0;
    padding: 0;
}

.cover-page {
    text-align: center;
    page-break-after: always;
}

.cover-page img {
    max-width: 100%;
    height: auto;
}

.title-page {
    text-align: center;
    page-break-after: always;
    padding: 3em 2em;
}

.title-page h1 {
    font-size: 2.5em;
    margin-bottom: 0.5em;
    font-weight: normal;
}

.title-page .author {
    font-size: 1.5em;
    margin-top: 2em;
    font-style: italic;
}

.part-title {
    font-size: 2em;
    font-weight: bold;
    text-align: center;
    margin: 2em 0 1em 0;
    page-break-before: always;
}

.epigraph {
    font-style: italic;
    text-align: right;
    margin: 2em 3em 2em 3em;
    padding: 1em;
    border-left: 3px solid #ccc;
    font-size: 0.95em;
    line-height: 1.8;
}

.epigraph-author {
    margin-top: 0.5em;
    text-align: right;
    font-weight: normal;
    font-size: 0.9em;
}

.chapter-title {
    font-size: 1.8em;
    font-weight: bold;
    text-align: center;
    margin: 2em 0 1em 0;
    page-break-before: always;
}

.text {
    text-align: justify;
    text-indent: 1.5em;
    margin: 0;
    line-height: 1.8;
}

.text:first-child {
    text-indent: 0;
}

.scene-break {
    text-align: center;
    margin: 1.5em 0;
    font-weight: normal;
    text-indent: 0;
    line-height: 1.8;
}

p {
    margin: 0;
}
"""


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load configuration from config.json file."""
    if config_path is None:
        config_path = Path(__file__).parent / "config.json"
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def format_text_to_html(content: str) -> str:
    """Convert plain text content to HTML paragraphs."""
    if not content:
        return ""
    
    paragraphs = content.split('\n\n')
    html_paragraphs = []
    
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        
        para_clean = para.replace(' ', '').replace('\n', '')
        if para_clean == '***' or para.strip() == '* * *' or (len(para.strip()) <= 10 and para.strip().count('*') >= 3):
            html_paragraphs.append('<p class="scene-break">***</p>')
            continue
        
        para = html.escape(para)
        para = para.replace('\n', '<br/>')
        html_paragraphs.append(f'<p class="text">{para}</p>')
    
    return '\n'.join(html_paragraphs)


def create_html_page(title: str, body_content: str, css_ref: str = "style/nav.css") -> str:
    """Create a complete HTML page with standard structure."""
    return f'''<?xml version="1.0" encoding="utf-8"?>
<!DOCTYPE html>
<html xmlns="http://www.w3.org/1999/xhtml">
<head>
    <title>{html.escape(title)}</title>
    <link rel="stylesheet" type="text/css" href="{css_ref}"/>
</head>
<body>
    {body_content}
</body>
</html>'''


def create_epub_item(book: epub.EpubBook, title: str, file_name: str, 
                     body_content: str, nav_css: epub.EpubItem) -> epub.EpubHtml:
    """Create and configure an EpubHtml item."""
    html_content = create_html_page(title, body_content)
    page = epub.EpubHtml(title=title, file_name=file_name, lang='ru')
    page.set_content(html_content.encode('utf-8'))
    page.add_item(nav_css)
    book.add_item(page)
    return page


def add_metadata(book: epub.EpubBook, data: Dict[str, Any]) -> None:
    """Set EPUB metadata from JSON data."""
    metadata = data.get('metadata', {})
    book.set_identifier(data.get('book_id', 'book_001'))
    book.set_title(metadata.get('title', 'Untitled'))
    book.set_language('ru')
    book.add_author(metadata.get('author', 'Unknown Author'))


def add_cover_image(book: epub.EpubBook, cover_path: str) -> Optional[str]:
    """Add cover image to EPUB and return cover filename."""
    if not os.path.exists(cover_path):
        return None
    
    with open(cover_path, 'rb') as f:
        cover_image = f.read()
    
    ext = os.path.splitext(cover_path)[1].lower()
    mime_type = MIME_TYPES.get(ext, 'image/jpeg')
    cover_name = f'images/cover{ext}'
    
    cover_item = epub.EpubItem(
        uid='cover-image',
        file_name=cover_name,
        media_type=mime_type,
        content=cover_image
    )
    book.add_item(cover_item)
    return cover_name


def create_css(book: epub.EpubBook) -> epub.EpubItem:
    """Create and add CSS stylesheet to EPUB."""
    nav_css = epub.EpubItem(
        uid="nav_css",
        file_name="style/nav.css",
        media_type="text/css",
        content=CSS_CONTENT
    )
    book.add_item(nav_css)
    return nav_css


def add_cover_page(book: epub.EpubBook, cover_name: str, nav_css: epub.EpubItem) -> epub.EpubHtml:
    """Create cover page with image."""
    body_content = f'<div class="cover-page"><img src="{cover_name}" alt="Обложка"/></div>'
    return create_epub_item(book, 'Обложка', 'cover.xhtml', body_content, nav_css)


def add_title_page(book: epub.EpubBook, metadata: Dict[str, Any], nav_css: epub.EpubItem) -> epub.EpubHtml:
    """Create title page with book title and author."""
    title = html.escape(metadata.get('title', 'Untitled'))
    author = html.escape(metadata.get('author', 'Unknown Author'))
    body_content = f'''<div class="title-page">
        <h1>{title}</h1>
        <div class="author">{author}</div>
    </div>'''
    return create_epub_item(book, 'Титульная страница', 'title.xhtml', body_content, nav_css)


def create_part_page(book: epub.EpubBook, part_idx: int, part_title: str, 
                     blockquote: Dict[str, str], nav_css: epub.EpubItem) -> epub.EpubHtml:
    """Create a part page with title and optional epigraph."""
    body_content = f'<h1 class="part-title">{html.escape(part_title)}</h1>'
    
    if blockquote:
        epigraph_text = blockquote.get('text', '')
        epigraph_author = blockquote.get('author', '')
        if epigraph_text:
            body_content += f'\n    <div class="epigraph">\n        <p>{html.escape(epigraph_text)}</p>'
            if epigraph_author:
                body_content += f'\n        <p class="epigraph-author">— {html.escape(epigraph_author)}</p>'
            body_content += '\n    </div>'
    
    file_name = f'part{part_idx:02d}.xhtml'
    return create_epub_item(book, part_title, file_name, body_content, nav_css)


def create_chapter_page(book: epub.EpubBook, part_idx: int, chapter: Dict[str, Any], 
                        nav_css: epub.EpubItem) -> epub.EpubHtml:
    """Create a chapter page with formatted content."""
    chapter_id = chapter.get('chapter_id', '')
    chapter_title = chapter.get('title', '')
    chapter_content = chapter.get('content', '')
    
    formatted_content = format_text_to_html(chapter_content)
    body_content = f'<h1 class="chapter-title">{html.escape(chapter_title)}</h1>\n    {formatted_content}'
    
    file_name = f'part{part_idx:02d}_{chapter_id}.xhtml'
    return create_epub_item(book, chapter_title, file_name, body_content, nav_css)


def normalize_data_structure(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Normalize JSON data to always have parts structure.
    Supports both formats:
    - {"parts": [...]} - structured with parts
    - {"chapters": [...]} - flat structure, creates virtual part
    """
    if 'parts' in data:
        return data['parts']
    elif 'chapters' in data:
        return [{
            'part_number': 1,
            'part_title': '',
            'blockquote': {},
            'chapters': data['chapters']
        }]
    return []


def process_content(book: epub.EpubBook, data: Dict[str, Any], 
                    nav_css: epub.EpubItem) -> Tuple[List, List]:
    """Process parts and chapters, return spine and TOC."""
    spine = []
    toc = []
    
    parts = normalize_data_structure(data)
    
    for part_idx, part in enumerate(parts, 1):
        part_title = part.get('part_title', '')
        blockquote = part.get('blockquote', {})
        chapters = part.get('chapters', [])
        
        if part_title:
            part_page = create_part_page(book, part_idx, part_title, blockquote, nav_css)
            spine.append(part_page)
            part_toc_items = []
        else:
            part_page = None
            part_toc_items = []
        
        for chapter in chapters:
            chapter_page = create_chapter_page(book, part_idx, chapter, nav_css)
            spine.append(chapter_page)
            part_toc_items.append(chapter_page)
        
        if part_page:
            toc.append((part_page, part_toc_items))
        else:
            toc.extend(part_toc_items)
    
    return spine, toc


def create_epub_from_json(json_path: str, cover_path: str, output_path: str) -> None:
    """Create EPUB file from JSON data."""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    book = epub.EpubBook()
    
    add_metadata(book, data)
    nav_css = create_css(book)
    
    cover_name = add_cover_image(book, cover_path)
    
    spine = ['nav']
    
    if cover_name:
        cover_page = add_cover_page(book, cover_name, nav_css)
        spine.append(cover_page)
    
    title_page = add_title_page(book, data.get('metadata', {}), nav_css)
    spine.append(title_page)
    
    content_spine, toc = process_content(book, data, nav_css)
    spine.extend(content_spine)
    
    book.spine = spine
    book.toc = toc
    
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())
    
    epub.write_epub(output_path, book, {})
    print(f"EPUB file created successfully: {output_path}")


if __name__ == '__main__':
    from config_helper import get_paths
    
    try:
        paths = get_paths()
        
        print(f"Creating EPUB for: {paths.book}")
        print(f"  Input JSON: {paths.translated_json}")
        print(f"  Cover: {paths.cover_image}")
        print(f"  Output: {paths.output_epub}")
        
        create_epub_from_json(
            str(paths.translated_json),
            str(paths.cover_image),
            str(paths.output_epub)
        )
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please check your config.json file")
        exit(1)
