#!/usr/bin/env python3
"""Сводный отчёт по бенчмарку: объединяет meta, diff_report и judge_report в report.md."""

import json
from pathlib import Path

BENCH_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BENCH_DIR / "results"


def load(p: Path):
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def total_cost(meta: dict) -> float:
    """OpenRouter возвращает usage.cost в долларах (если include usage)."""
    total = 0.0
    for stage in meta.get("stages", {}).values():
        u = stage.get("usage") or {}
        c = u.get("cost")
        if c is not None:
            try:
                total += float(c)
            except Exception:
                pass
    return total


def total_tokens(meta: dict):
    p, c = 0, 0
    for stage in meta.get("stages", {}).values():
        u = stage.get("usage") or {}
        p += int(u.get("prompt_tokens", 0) or 0)
        c += int(u.get("completion_tokens", 0) or 0)
    return p, c


def total_latency(meta: dict) -> float:
    return sum(float(s.get("latency_sec", 0) or 0) for s in meta.get("stages", {}).values())


def avg_judge(per_judge: dict):
    scores = []
    for judge_result in (per_judge or {}).values():
        if judge_result and "scores" in judge_result:
            vals = list(judge_result["scores"].values())
            if vals:
                scores.append(sum(vals) / len(vals))
    if not scores:
        return None, None
    avg = sum(scores) / len(scores)
    spread = max(scores) - min(scores) if len(scores) > 1 else 0
    return avg, spread


def main():
    diff = load(BENCH_DIR / "diff_report.json") or {}
    judge = load(BENCH_DIR / "judge_report.json") or {}

    rows = []
    for model_dir in sorted(RESULTS_DIR.iterdir()):
        if not model_dir.is_dir():
            continue
        meta = load(model_dir / "meta.json") or {}
        slug = meta.get("model", model_dir.name)
        label = meta.get("label", slug)

        p, c = total_tokens(meta)
        cost = total_cost(meta)
        latency = total_latency(meta)
        diff_row = diff.get(slug, {})
        jdg_avg, jdg_spread = avg_judge(judge.get(slug))

        rows.append({
            "label": label,
            "slug": slug,
            "coverage_pct": diff_row.get("coverage_pct"),
            "translation_match_pct": diff_row.get("translation_match_pct"),
            "json_valid": diff_row.get("json_valid"),
            "judge_avg": jdg_avg,
            "judge_spread": jdg_spread,
            "tokens_in": p,
            "tokens_out": c,
            "cost_usd": cost,
            "latency_sec": latency,
        })

    rows.sort(key=lambda r: (-(r["judge_avg"] or 0), -(r["coverage_pct"] or 0)))

    lines = []
    lines.append("# Benchmark: Glossary + Styleguide Generation\n")
    lines.append("Книга: Dark Age. Эталон: Gemini 3.1 Pro (`__OUT/Dark Age/`).\n\n")
    lines.append("## Сводная таблица\n")
    lines.append("| # | Модель | Coverage % | Tr.match % | JSON | Judge avg | Δ судей | Tokens in/out | Стоимость $ | Время, с |")
    lines.append("|---|--------|-----------:|-----------:|:----:|----------:|--------:|:-------------:|------------:|--------:|")
    for i, r in enumerate(rows, 1):
        lines.append(
            f"| {i} | {r['label']} | "
            f"{r['coverage_pct'] if r['coverage_pct'] is not None else '—'} | "
            f"{r['translation_match_pct'] if r['translation_match_pct'] is not None else '—'} | "
            f"{'✓' if r['json_valid'] else '✗'} | "
            f"{r['judge_avg']:.2f}" + (" | " if r['judge_avg'] is not None else "— | ") +
            (f"{r['judge_spread']:.1f}" if r['judge_spread'] is not None else "—") + " | "
            f"{r['tokens_in']}/{r['tokens_out']} | "
            f"{r['cost_usd']:.3f} | "
            f"{r['latency_sec']:.0f} |"
        )

    lines.append("\n## Детали по моделям\n")
    for r in rows:
        lines.append(f"### {r['label']} (`{r['slug']}`)\n")
        diff_row = diff.get(r["slug"], {})
        if diff_row:
            lines.append("**Diff vs эталон:**")
            lines.append("```json")
            lines.append(json.dumps(diff_row, ensure_ascii=False, indent=2))
            lines.append("```")
        judge_row = judge.get(r["slug"]) or {}
        for judge_id, j in judge_row.items():
            if not j:
                continue
            lines.append(f"\n**Судья `{judge_id}`:**")
            lines.append("```json")
            lines.append(json.dumps(j, ensure_ascii=False, indent=2))
            lines.append("```")
        lines.append("")

    out = BENCH_DIR / "report.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"✓ Сохранено: {out}")


if __name__ == "__main__":
    main()
