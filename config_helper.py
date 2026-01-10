#!/usr/bin/env python3
"""
Helper module for working with optimized config.json structure.
Provides path generation functions based on book name and settings.
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional


def load_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """Load configuration from config.json file."""
    if config_path is None:
        config_path = Path(__file__).parent / "config.json"
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)


class ConfigPaths:
    """Helper class to generate all file paths from optimized config."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize with config dict or load from default location."""
        if config is None:
            config = load_config()
        
        self.config = config
        self.book = config['book']
        self.paths = config['paths']
        self.settings = config['settings']
        
        # Base directories
        self.input_dir = Path(self.paths['input_dir'])
        self.output_dir = Path(self.paths['output_dir'])
        self.work_dir = self.output_dir / self.book
    
    # Input files
    @property
    def epub_source(self) -> Path:
        """Path to source EPUB file."""
        return self.input_dir / self.paths['epub_file']
    
    @property
    def cover_image(self) -> Path:
        """Path to cover image. Supports both relative and absolute paths."""
        cover_path = self.paths['cover_image']
        # If path is absolute, use as-is
        if cover_path.startswith('/'):
            return Path(cover_path)
        # If path starts with ./ or ../, use as-is (relative to current dir)
        if cover_path.startswith('./') or cover_path.startswith('../'):
            return Path(cover_path)
        # If path starts with __IN, __OUT, or other __ prefix, use as-is (project-relative)
        if cover_path.startswith('__'):
            return Path(cover_path)
        # Otherwise, treat as relative to input_dir
        return self.input_dir / cover_path
    
    # Output files (in work directory)
    @property
    def output_json(self) -> Path:
        """Path to extracted JSON from EPUB."""
        return self.work_dir / f"{self.book}.json"
    
    @property
    def glossary_json(self) -> Path:
        """Path to glossary JSON file."""
        return self.work_dir / f"{self.book}_glossary.json"
    
    @property
    def styleguide_md(self) -> Path:
        """Path to style guide markdown file."""
        return self.work_dir / f"{self.book}_styleguide.md"
    
    @property
    def translated_json(self) -> Path:
        """Path to translated JSON file."""
        return self.work_dir / f"{self.book}_ru.json"
    
    @property
    def checkpoint_json(self) -> Path:
        """Path to translation checkpoint file."""
        return self.work_dir / "translation_checkpoint.json"
    
    @property
    def output_epub(self) -> Path:
        """Path to final translated EPUB file."""
        return self.output_dir / f"{self.book}_ru.epub"
    
    # Settings
    @property
    def translation_max_tokens(self) -> int:
        """Maximum tokens per translation batch."""
        return self.settings['translation_max_tokens']
    
    @property
    def translation_context(self) -> int:
        """Number of context paragraphs for translation."""
        return self.settings['translation_context']
    
    def ensure_work_dir(self) -> None:
        """Create work directory if it doesn't exist."""
        self.work_dir.mkdir(parents=True, exist_ok=True)
    
    def __repr__(self) -> str:
        """String representation for debugging."""
        return f"ConfigPaths(book='{self.book}', work_dir='{self.work_dir}')"


def get_paths() -> ConfigPaths:
    """Convenience function to get ConfigPaths instance."""
    return ConfigPaths()


if __name__ == '__main__':
    # Test the helper
    paths = get_paths()
    print(f"Book: {paths.book}")
    print(f"EPUB source: {paths.epub_source}")
    print(f"Output JSON: {paths.output_json}")
    print(f"Glossary: {paths.glossary_json}")
    print(f"Style guide: {paths.styleguide_md}")
    print(f"Translated JSON: {paths.translated_json}")
    print(f"Checkpoint: {paths.checkpoint_json}")
    print(f"Cover: {paths.cover_image}")
    print(f"Output EPUB: {paths.output_epub}")
    print(f"\nSettings:")
    print(f"  Translation max tokens: {paths.translation_max_tokens}")
    print(f"  Translation context: {paths.translation_context}")
