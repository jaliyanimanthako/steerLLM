#!/usr/bin/env python3
"""Build a reproducible 200-question pilot dataset for complexity steering.

Sources (train splits only):
  * EntailmentBank task1: 100 questions + supporting facts
  * KILT-ELI5 dev:         70 long-form questions + Wikipedia provenance refs
  * QASPER train:          30 questions + annotated evidence

Outputs: questions_200.csv, questions_200.jsonl, dataset_summary.json
Requires: pip install -U datasets requests
"""

import argparse
import csv
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path

import requests
from datasets import load_dataset

KILT_URL = "https://dl.fbaipublicfiles.com/KILT/eli5-dev-kilt.jsonl"
TARGETS = {"EntailmentBank": 100, "KILT-ELI5": 70, "QASPER": 30}
EXPLANATORY = re.compile(r"\b(how|why|explain|what causes|what happens|in what way|what is the difference|what are the implications)\b", re.I)


def clean(value):
    return re.sub(r"\s+", " ", str(value or "")).strip()


def is_explanatory(question):
    return bool(EXPLANATORY.search(question)) or len(question.split()) >= 12


def order_candidates(rows, rng):
    rng.shuffle(rows)
    # Favor questions suited to explanations but retain enough candidates to fill quotas.
    return sorted(rows, key=lambda r: bool(is_explanatory(r["question"])), reverse=True)


def entailment_candidates(rng):
    ds = load_dataset("sxiong/entailmentbank", "task1", split="train")
    rows = []
    for item in ds:
        meta = item.get("meta") or {}
        triples = meta.get("triples") or {}
        # Gold task1 context contains only relevant candidate facts.
        if isinstance(triples, dict) and triples:
            context = "\n".join(f"{k}: {v}" for k, v in sorted(triples.items()))
        else:
            context = clean(item.get("context"))
        q = clean(item.get("question"))
        if not q or not context:
            continue
        rows.append(dict(source="EntailmentBank", source_id=str(item.get("id", "")),
                         question=q, context=context, context_status="provided_supporting_facts",
                         reference_answer=clean(item.get("answer")),
                         evidence_references="", source_split="train"))
    return order_candidates(rows, rng)


def kilt_candidates(rng):
    # Small official dev split (~14 MB); avoids downloading 548 MB train file.
    response = requests.get(KILT_URL, timeout=120)
    response.raise_for_status()
    rows = []
    for line in response.text.splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        q = clean(item.get("input"))
        outputs = item.get("output") or []
        answer = next((clean(o.get("answer")) for o in outputs if clean(o.get("answer"))), "")
        provenance = []
        for o in outputs:
            for p in o.get("provenance") or []:
                provenance.append({k: p.get(k) for k in
                                   ("wikipedia_id", "title", "section", "start_paragraph_id", "end_paragraph_id")
                                   if p.get(k) is not None})
        if not q or len(q.split()) < 5:
            continue
        rows.append(dict(source="KILT-ELI5", source_id=str(item.get("id", "")),
                         question=q, context="", context_status="provenance_only_retrieval_required",
                         reference_answer=answer,
                         evidence_references=json.dumps(provenance, ensure_ascii=False),
                         source_split="dev"))
    return order_candidates(rows, rng)


def qasper_candidates(rng):
    ds = load_dataset("allenai/qasper", split="train")
    rows = []
    for paper in ds:
        qas = paper.get("qas") or {}
        questions = qas.get("question") or []
        answers_groups = qas.get("answers") or []
        for idx, raw_q in enumerate(questions):
            q = clean(raw_q)
            if not q or idx >= len(answers_groups):
                continue
            group = answers_groups[idx] or {}
            annotations = group.get("answer") or []
            evidence, reference = [], ""
            for ann in annotations:
                if ann.get("unanswerable") or ann.get("yes_no") is not None:
                    # Prefer non-yes/no explanatory questions; QASPER may encode yes_no as None.
                    if ann.get("unanswerable") or isinstance(ann.get("yes_no"), bool):
                        continue
                evidence.extend(clean(x) for x in (ann.get("evidence") or [])
                                if isinstance(x, str) and not x.startswith("FLOAT SELECTED"))
                if not reference:
                    reference = clean(ann.get("free_form_answer")) or "; ".join(
                        clean(x) for x in (ann.get("extractive_spans") or []))
            # Keep only records with actual provided evidence, not just an abstract.
            evidence = list(dict.fromkeys(e for e in evidence if len(e.split()) >= 7))
            if not evidence or not reference:
                continue
            # Limit very long passages while preserving explicit truncation indicator.
            context = "\n\n".join(evidence)
            words = context.split()
            if len(words) > 450:
                context = " ".join(words[:450])
                status = "annotated_evidence_truncated_450_words"
            else:
                status = "annotated_evidence"
            rows.append(dict(source="QASPER", source_id=f"{paper['id']}:q{idx}",
                             question=q, context=context, context_status=status,
                             reference_answer=reference, evidence_references="",
                             source_split="train"))
    return order_candidates(rows, rng)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="steering_questions")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    rng = random.Random(args.seed)
    sources = [entailment_candidates, kilt_candidates, qasper_candidates]
    selected, seen_questions = [], set()
    for fn in sources:
        candidates = fn(rng)
        source_name = candidates[0]["source"] if candidates else fn.__name__
        collected = 0
        for row in candidates:
            canonical = re.sub(r"\W+", "", row["question"].lower())
            digest = hashlib.sha256(canonical.encode()).hexdigest()
            if digest in seen_questions:
                continue
            seen_questions.add(digest)
            selected.append(row)
            collected += 1
            if collected == TARGETS[source_name]:
                break
        if collected != TARGETS[source_name]:
            raise RuntimeError(f"Could only select {collected}/{TARGETS[source_name]} from {source_name}. "
                               "Consider relaxing evidence/question filters; no partial result saved.")
        print(f"{source_name}: {collected} questions")

    # Assign deterministic identifiers and question-group splits; no paired answers yet.
    rng.shuffle(selected)
    for i, row in enumerate(selected):
        row["id"] = f"Q{i+1:03d}"
        row["research_split"] = "discovery" if i < 120 else "validation" if i < 160 else "test"
        row["beginner_answer"] = ""
        row["expert_answer"] = ""

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    fields = ["id", "source", "source_id", "source_split", "research_split",
              "question", "context", "context_status", "reference_answer",
              "evidence_references", "beginner_answer", "expert_answer"]
    with (output / "questions_200.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(selected)
    with (output / "questions_200.jsonl").open("w", encoding="utf-8") as fh:
        for row in selected:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {"seed": args.seed, "total": len(selected),
               "sources": dict(Counter(r["source"] for r in selected)),
               "research_splits": dict(Counter(r["research_split"] for r in selected)),
               "context_status": dict(Counter(r["context_status"] for r in selected))}
    (output / "dataset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Saved to {output.resolve()}")


if __name__ == "__main__":
    main()
