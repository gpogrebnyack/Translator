#!/usr/bin/env python3
"""
Бенчмарк-раннер: для каждой модели из bench_config.json делает два вызова
(глоссарий + стайлгайд) через OpenRouter, сохраняет выходы и метаданные.
"""

import argparse
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

# Загружаем отдельный bench/.env (не мешаем основному .env проекта)
load_dotenv(BENCH_DIR / ".env")

from importlib import import_module  # noqa: E402
_pipeline = import_module("1_glossary_and_styleguide")
prepare_book_text = _pipeline.prepare_book_text
load_book = _pipeline.load_book
extract_json_from_response = _pipeline.extract_json_from_response

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MODELS_URL = "https://openrouter.ai/api/v1/models"

GLOSSARY_SYS = "Ты — ведущий редактор и специалист по локализации художественной литературы."
STYLEGUIDE_SYS = "Ты — эксперт по литературному переводу и стилистическому анализу текстов."


def sanitize(slug: str) -> str:
    return slug.replace("/", "__").replace(":", "_")


def load_config():
    with open(BENCH_DIR / "bench_config.json", "r", encoding="utf-8") as f:
        return json.load(f)


def load_prompt(rel_path: str) -> str:
    with open(ROOT / rel_path, "r", encoding="utf-8") as f:
        return f.read()


def validate_slugs(candidates, api_key):
    """Проверяет, что все slug-и существуют на OpenRouter. Возвращает dict id->context_length."""
    print("▶ Валидация slug-ов через OpenRouter /models ...")
    try:
        r = requests.get(MODELS_URL, headers={"Authorization": f"Bearer {api_key}"}, timeout=30)
        r.raise_for_status()
        data = r.json().get("data", [])
        available = {m["id"]: m for m in data}
    except Exception as e:
        print(f"⚠ Не удалось получить список моделей: {e}. Пропускаю валидацию.")
        return {}

    info = {}
    for cand in candidates:
        slug = cand["id"]
        if slug in available:
            ctx = available[slug].get("context_length")
            info[slug] = ctx
            print(f"  ✓ {slug} (context: {ctx})")
        else:
            print(f"  ✗ {slug} — НЕ НАЙДЕН на OpenRouter")
            info[slug] = None
    return info


def call_api(model: str, system: str, user: str, api_key: str, *, temperature: float,
             max_tokens: int, timeout: int, max_retries: int):
    """Возвращает (content, usage_dict, latency_sec, status_code)."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/translator-app",
        "X-Title": "Translator Bench",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "usage": {"include": True},
    }

    last_err = None
    for attempt in range(1, max_retries + 1):
        t0 = time.time()
        try:
            r = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=timeout)
            latency = time.time() - t0
            if r.status_code == 200:
                data = r.json()
                content = data["choices"][0]["message"]["content"]
                usage = data.get("usage", {})
                return content, usage, latency, r.status_code
            last_err = f"HTTP {r.status_code}: {r.text[:500]}"
            print(f"  ✗ Попытка {attempt}/{max_retries}: {last_err}")
        except Exception as e:
            last_err = str(e)
            print(f"  ✗ Попытка {attempt}/{max_retries}: {last_err}")
        if attempt < max_retries:
            wait = attempt * 10
            print(f"  ⏱  Жду {wait}с перед ретраем...")
            time.sleep(wait)
    return None, {}, 0.0, 0


def run_candidate(cand, book_text, prompt_glossary, prompt_styleguide, api_cfg, api_key, out_dir: Path):
    slug = cand["id"]
    label = cand["label"]
    print(f"\n{'=' * 60}\n▶ {label}  ({slug})\n{'=' * 60}")

    model_dir = out_dir / sanitize(slug)
    model_dir.mkdir(parents=True, exist_ok=True)

    meta = {"model": slug, "label": label, "stages": {}}

    # --- ЭТАП 1: ГЛОССАРИЙ ---
    glossary_user = f"{prompt_glossary}\n\n# ТЕКСТ КНИГИ:\n\n{book_text}"
    content, usage, latency, status = call_api(
        slug, GLOSSARY_SYS, glossary_user, api_key,
        temperature=api_cfg["glossary_temperature"],
        max_tokens=api_cfg["max_tokens"],
        timeout=api_cfg["timeout"],
        max_retries=api_cfg["max_retries"],
    )
    if content is None:
        meta["stages"]["glossary"] = {"status": "failed", "http_status": status}
        with open(model_dir / "meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        print(f"✗ Глоссарий провалился, пропускаю стайлгайд для {slug}")
        return meta

    # Сырой ответ
    (model_dir / "glossary_raw.txt").write_text(content, encoding="utf-8")
    glossary = extract_json_from_response(content)
    json_valid = isinstance(glossary, dict)
    if json_valid:
        with open(model_dir / "glossary.json", "w", encoding="utf-8") as f:
            json.dump(glossary, f, ensure_ascii=False, indent=2)
    else:
        (model_dir / "glossary.json").write_text(content, encoding="utf-8")

    meta["stages"]["glossary"] = {
        "status": "ok",
        "http_status": status,
        "latency_sec": round(latency, 2),
        "json_valid": json_valid,
        "usage": usage,
    }
    print(f"  ✓ Глоссарий: {usage.get('prompt_tokens')}p / {usage.get('completion_tokens')}c токенов, "
          f"{latency:.1f}с, json_valid={json_valid}")

    # --- ЭТАП 2: СТАЙЛГАЙД ---
    glossary_str = json.dumps(glossary, ensure_ascii=False, indent=2) if json_valid else content
    styleguide_user = (
        f"{prompt_styleguide}\n\n# ГЛОССАРИЙ:\n\n```json\n{glossary_str}\n```\n\n"
        f"# ТЕКСТ КНИГИ:\n\n{book_text}"
    )
    sg_content, sg_usage, sg_latency, sg_status = call_api(
        slug, STYLEGUIDE_SYS, styleguide_user, api_key,
        temperature=api_cfg["styleguide_temperature"],
        max_tokens=api_cfg["max_tokens"],
        timeout=api_cfg["timeout"],
        max_retries=api_cfg["max_retries"],
    )
    if sg_content is None:
        meta["stages"]["styleguide"] = {"status": "failed", "http_status": sg_status}
    else:
        (model_dir / "styleguide.md").write_text(sg_content, encoding="utf-8")
        meta["stages"]["styleguide"] = {
            "status": "ok",
            "http_status": sg_status,
            "latency_sec": round(sg_latency, 2),
            "usage": sg_usage,
        }
        print(f"  ✓ Стайлгайд: {sg_usage.get('prompt_tokens')}p / {sg_usage.get('completion_tokens')}c токенов, "
              f"{sg_latency:.1f}с")

    with open(model_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return meta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", help="Запустить только одну модель (slug)")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Пропустить модели, у которых уже есть meta.json")
    parser.add_argument("--dry-run", action="store_true",
                        help="Только валидация slug-ов и подсчёт токенов, без API-вызовов")
    args = parser.parse_args()

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("✗ OPENROUTER_API_KEY не найден. Проверь bench/.env")
        sys.exit(1)

    cfg = load_config()
    book_data = load_book(str(ROOT / cfg["book_json"]))
    book_text = prepare_book_text(book_data)
    # Грубая оценка токенов: ~4 chars/token
    approx_tokens = len(book_text) // 4
    print(f"Книга: {len(book_text)} символов, ~{approx_tokens} токенов")

    prompt_glossary = load_prompt(cfg["prompts"]["glossary"])
    prompt_styleguide = load_prompt(cfg["prompts"]["styleguide"])

    candidates = cfg["candidates"]
    if args.only:
        candidates = [c for c in candidates if c["id"] == args.only]
        if not candidates:
            print(f"✗ Модель {args.only} не найдена в bench_config.json")
            sys.exit(1)

    ctx_info = validate_slugs(candidates, api_key)

    if args.dry_run:
        print("\n--- DRY RUN: вход оценен, вызовы не делаются ---")
        return

    out_dir = BENCH_DIR / "results"
    out_dir.mkdir(exist_ok=True)

    api_cfg = cfg["api"]
    for cand in candidates:
        model_dir = out_dir / sanitize(cand["id"])
        if args.skip_existing and (model_dir / "meta.json").exists():
            print(f"⏭  {cand['id']} — уже есть meta.json, пропускаю")
            continue
        try:
            run_candidate(cand, book_text, prompt_glossary, prompt_styleguide,
                          api_cfg, api_key, out_dir)
        except KeyboardInterrupt:
            print("\n⚠ Прервано пользователем")
            raise
        except Exception as e:
            print(f"✗ Ошибка на {cand['id']}: {e}")
            import traceback
            traceback.print_exc()

    print("\n✓ Бенчмарк завершён. Следующий шаг: bench/diff_vs_reference.py и bench/judge.py")


if __name__ == "__main__":
    main()
