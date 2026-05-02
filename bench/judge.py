#!/usr/bin/env python3
"""
LLM-as-judge: два судьи (GPT-5.4 Pro + Claude Opus 4.7) оценивают
глоссарий+стайлгайд каждой модели-кандидата по рубрике 1..10.
"""

import json
import os
import sys
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
BENCH_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

load_dotenv(BENCH_DIR / ".env")

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

JUDGE_SYSTEM = (
    "Ты — строгий и опытный литературный редактор русского перевода. "
    "Твоя задача — объективно оценить качество глоссария и стайлгайда, "
    "подготовленных для перевода англоязычного романа на русский. "
    "Отвечай строго валидным JSON."
)

RUBRIC = """
Оцени по каждому критерию от 1 до 10 (1 — очень плохо, 10 — превосходно):

1. coverage: полнота глоссария — все ли ключевые персонажи, локации, термины охвачены?
2. translation_quality: корректность и литературность русских переводов имён/терминов. Штрафуй за бездумные кальки и ложных друзей переводчика.
3. consistency: внутренняя консистентность (одинаковые паттерны транслитерации имён, единообразие титулов).
4. format_adherence: соблюдение структуры из промпта (JSON c категориями characters/locations/items/organizations/terms, скобочные пояснения).
5. styleguide_depth: глубина анализа в стайлгайде (жанр, тон, матрица «ты/вы», голоса персонажей).
6. mandatory_section_preserved: сохранён ли дословно раздел «2. Linguistics & Adaptation» из мета-промпта. 10 если дословно, 1 если удалён/переписан.
7. overall_usefulness: итоговая применимость для живого переводчика.

Ответ строго в JSON:
{
  "scores": {"coverage": N, "translation_quality": N, "consistency": N,
             "format_adherence": N, "styleguide_depth": N,
             "mandatory_section_preserved": N, "overall_usefulness": N},
  "strong_points": ["..."],
  "weak_points": ["..."],
  "top_issues": ["конкретные примеры плохих переводов или пропусков"]
}
"""


def sanitize(slug: str) -> str:
    return slug.replace("/", "__").replace(":", "_")


def load_cfg():
    return json.loads((BENCH_DIR / "bench_config.json").read_text(encoding="utf-8"))


def load_book_sample(book_text_path: Path, max_chars: int = 40000) -> str:
    """Берём первые max_chars символов книги — этого хватит судье для контекста."""
    text = book_text_path.read_text(encoding="utf-8") if book_text_path.exists() else ""
    return text[:max_chars]


def build_judge_prompt(prompt_glossary, prompt_styleguide, reference_glossary,
                      candidate_glossary_raw, candidate_styleguide, book_sample):
    return f"""Ниже — материалы для оценки.

== ОРИГИНАЛЬНЫЙ ПРОМПТ ДЛЯ ГЛОССАРИЯ ==
{prompt_glossary}

== ОРИГИНАЛЬНЫЙ ПРОМПТ ДЛЯ СТАЙЛГАЙДА ==
{prompt_styleguide}

== ФРАГМЕНТ КНИГИ (для контекста) ==
{book_sample}

== ЭТАЛОН (Gemini 3.1 Pro, как ориентир, НЕ обязательная истина) ==
```json
{reference_glossary}
```

== ОЦЕНИВАЕМЫЙ ГЛОССАРИЙ КАНДИДАТА ==
{candidate_glossary_raw}

== ОЦЕНИВАЕМЫЙ СТАЙЛГАЙД КАНДИДАТА ==
{candidate_styleguide}

{RUBRIC}
"""


def call_judge(model, system, user, api_key, max_retries=3):
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/translator-app",
        "X-Title": "Translator Bench Judge",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": 0.0,
        "max_tokens": 8000,
        "reasoning": {"max_tokens": 2000},
    }
    for attempt in range(1, max_retries + 1):
        try:
            r = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=600)
            if r.status_code == 200:
                return r.json()["choices"][0]["message"]["content"]
            print(f"  ✗ judge {model} attempt {attempt}: HTTP {r.status_code}")
        except Exception as e:
            print(f"  ✗ judge {model} attempt {attempt}: {e}")
        time.sleep(attempt * 10)
    return None


def extract_json_block(text: str):
    if text is None:
        return None
    s = text.strip()
    if "```json" in s:
        s = s.split("```json", 1)[1].split("```", 1)[0].strip()
    elif "```" in s:
        s = s.split("```", 1)[1].split("```", 1)[0].strip()
    try:
        return json.loads(s)
    except Exception:
        # Попробуем найти первую фигурную скобку
        start = s.find("{")
        end = s.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(s[start:end + 1])
            except Exception:
                return None
    return None


def main():
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("✗ OPENROUTER_API_KEY не найден. bench/.env")
        sys.exit(1)

    cfg = load_cfg()
    prompt_glossary = (ROOT / cfg["prompts"]["glossary"]).read_text(encoding="utf-8")
    prompt_styleguide = (ROOT / cfg["prompts"]["styleguide"]).read_text(encoding="utf-8")
    reference_glossary = (ROOT / cfg["reference_glossary"]).read_text(encoding="utf-8")

    # Книжный фрагмент: грузим prepare_book_text и берём начало
    from importlib import import_module
    _pipeline = import_module("1_glossary_and_styleguide")
    book_data = _pipeline.load_book(str(ROOT / cfg["book_json"]))
    book_text = _pipeline.prepare_book_text(book_data)
    book_sample = book_text[:40000]

    results_dir = BENCH_DIR / "results"
    out = {}

    for model_dir in sorted(results_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        meta_path = model_dir / "meta.json"
        if not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        slug = meta.get("model", model_dir.name)

        gloss_raw = (model_dir / "glossary_raw.txt")
        sg_md = (model_dir / "styleguide.md")
        if not gloss_raw.exists() or not sg_md.exists():
            print(f"⏭  {slug}: нет полного набора файлов, пропускаю")
            continue

        cand_gloss = gloss_raw.read_text(encoding="utf-8")
        cand_sg = sg_md.read_text(encoding="utf-8")

        user_prompt = build_judge_prompt(
            prompt_glossary, prompt_styleguide, reference_glossary,
            cand_gloss, cand_sg, book_sample
        )

        per_judge = {}
        for judge in cfg["judges"]:
            print(f"\n▶ Судья {judge['id']} оценивает {slug} ...")
            raw = call_judge(judge["id"], JUDGE_SYSTEM, user_prompt, api_key)
            parsed = extract_json_block(raw) if raw else None
            (model_dir / f"judge_{sanitize(judge['id'])}_raw.txt").write_text(raw or "", encoding="utf-8")
            per_judge[judge["id"]] = parsed
            if parsed and "scores" in parsed:
                avg = sum(parsed["scores"].values()) / len(parsed["scores"])
                print(f"  средний балл: {avg:.2f}")

        (model_dir / "judge.json").write_text(
            json.dumps(per_judge, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        out[slug] = per_judge

    (BENCH_DIR / "judge_report.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n✓ judge_report.json сохранён")


if __name__ == "__main__":
    main()
