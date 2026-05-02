#!/usr/bin/env python3
"""
Количественное сравнение глоссариев моделей с эталонным (Gemini 3.1 Pro).
Coverage, translation match, extras, category fidelity, JSON валидность.
"""

import json
from difflib import SequenceMatcher
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BENCH_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BENCH_DIR / "results"

REQUIRED_CATEGORIES = {"characters", "locations", "items", "organizations", "terms"}


def load_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def norm_key(k: str) -> str:
    return k.strip().lower()


def fuzzy_eq(a: str, b: str, thresh: float = 0.8) -> bool:
    if not a or not b:
        return False
    return SequenceMatcher(None, a.lower(), b.lower()).ratio() >= thresh


def flatten(gloss: dict):
    """dict of category -> {key: value} → dict of key_normalized -> (category, value)."""
    out = {}
    if not isinstance(gloss, dict):
        return out
    for cat, items in gloss.items():
        if not isinstance(items, dict):
            continue
        for k, v in items.items():
            if not isinstance(k, str):
                continue
            # Берём первую часть до " / " как канонический ключ
            key = norm_key(k.split(" / ")[0])
            out[key] = (cat, str(v))
    return out


def analyze(candidate_path: Path, reference: dict) -> dict:
    cand = load_json(candidate_path)
    if cand is None:
        return {"json_valid": False, "error": "не удалось распарсить JSON"}

    ref_flat = flatten(reference)
    cand_flat = flatten(cand)

    ref_keys = set(ref_flat.keys())
    cand_keys = set(cand_flat.keys())

    covered = ref_keys & cand_keys
    # fuzzy-совпадения для ключей: если канд. не имеет точного ключа, ищем похожий
    fuzzy_covered = set(covered)
    for rk in ref_keys - covered:
        for ck in cand_keys - covered:
            if fuzzy_eq(rk, ck, 0.85):
                fuzzy_covered.add(rk)
                break

    # Translation match на пересечении
    translation_matches = 0
    for k in covered:
        ref_val = ref_flat[k][1].split(" (")[0].strip()  # отбрасываем пояснение в скобках
        cand_val = cand_flat[k][1].split(" (")[0].strip()
        if fuzzy_eq(ref_val, cand_val, 0.85):
            translation_matches += 1

    cats = set(k for k in cand.keys() if isinstance(cand.get(k), dict))
    missing_cats = REQUIRED_CATEGORIES - cats
    # Если у модели другой набор (например factions_and_organizations вместо organizations), тоже помечаем
    extra_cats = cats - REQUIRED_CATEGORIES

    return {
        "json_valid": True,
        "ref_total": len(ref_keys),
        "cand_total": len(cand_keys),
        "covered_exact": len(covered),
        "covered_fuzzy": len(fuzzy_covered),
        "coverage_pct": round(100 * len(fuzzy_covered) / max(len(ref_keys), 1), 1),
        "translation_match": translation_matches,
        "translation_match_pct": round(100 * translation_matches / max(len(covered), 1), 1),
        "extras": len(cand_keys - ref_keys),
        "missing_required_categories": sorted(missing_cats),
        "extra_categories": sorted(extra_cats),
    }


def main():
    ref_path = ROOT / "__OUT" / "Dark Age" / "Dark Age_glossary.json"
    reference = load_json(ref_path)
    if reference is None:
        print(f"✗ Не удалось загрузить эталон: {ref_path}")
        return

    report = {}
    for model_dir in sorted(RESULTS_DIR.iterdir()):
        if not model_dir.is_dir():
            continue
        gloss_path = model_dir / "glossary.json"
        if not gloss_path.exists():
            continue
        meta_path = model_dir / "meta.json"
        meta = load_json(meta_path) or {}
        slug = meta.get("model", model_dir.name)

        result = analyze(gloss_path, reference)
        report[slug] = result
        print(f"\n▶ {slug}")
        for k, v in result.items():
            print(f"    {k}: {v}")

    out_path = BENCH_DIR / "diff_report.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n✓ Сохранено: {out_path}")


if __name__ == "__main__":
    main()
