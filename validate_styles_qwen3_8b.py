#!/usr/bin/env python3
"""Evaluate paired beginner/expert answers with local Qwen3-8B (inference only).

Input: CSV produced by create_steering_questions.py AFTER populating
       beginner_answer and expert_answer with synthetic teacher responses.
Output: per-pair scores CSV, JSONL and summary JSON.

Note: A same-family LLM judge is noisy; labels are not ground truth.
KILT-ELI5 rows without retrieved context cannot be assessed for source grounding.
"""
import argparse
import csv
import json
import random
import re
from collections import Counter
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SYSTEM = """You are a careful annotator evaluating paired explanations for research.
Return one valid JSON object, and no Markdown. Use the exact keys given.
Judge linguistic/explanatory complexity, not answer length or whether you personally prefer an answer.
Treat the provided context as the ONLY permitted factual evidence when it exists.
If context is missing, do NOT invent a groundedness score.
Assess both answers' core factual claims and note missing or added facts.
Score dimensions on integer 1..5: complexity (1=very accessible, 5=highly technical),
clarity (1=unclear, 5=clear), fact_alignment (1=major mismatch, 5=same core claims).
Groundedness is 1..5 if context exists, otherwise null.
You must use the output labels A and B, not guess which was designated beginner.
Required JSON keys: complexity_A, complexity_B, clarity_A, clarity_B,
fact_alignment, groundedness_A, groundedness_B, shared_facts_preserved,
unsupported_claims_A, unsupported_claims_B, explanation.
For shared_facts_preserved use boolean. For unsupported_claims_A/B use arrays of strings.
Explanation must be one brief sentence."""


def extract_json(text):
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        if start < 0:
            raise
        decoder = json.JSONDecoder()
        obj, _ = decoder.raw_decode(text[start:])
        return obj


def valid(j, has_context):
    keys = ("complexity_A", "complexity_B", "clarity_A", "clarity_B", "fact_alignment")
    if not isinstance(j, dict) or not all(type(j.get(k)) is int and 1 <= j[k] <= 5 for k in keys):
        return False
    if not isinstance(j.get("shared_facts_preserved"), bool):
        return False
    if not all(isinstance(j.get(k), list) for k in ("unsupported_claims_A", "unsupported_claims_B")):
        return False
    if has_context:
        if not all(type(j.get(k)) is int and 1 <= j[k] <= 5 for k in ("groundedness_A", "groundedness_B")):
            return False
    else:
        if j.get("groundedness_A") is not None or j.get("groundedness_B") is not None:
            return False
    return True


def build_prompt(row, a, b):
    context = (row.get("context") or "").strip()
    evidence = context if context else "[NOT AVAILABLE: judge pair consistency only; grounding cannot be verified.]"
    return (f"Question:\n{row['question']}\n\n"
            f"Factual source context:\n{evidence}\n\n"
            f"Response A:\n{a}\n\nResponse B:\n{b}\n\n"
            "Assess both responses and output JSON with all required keys.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="steering_questions/questions_200.csv")
    ap.add_argument("--output", default="style_validation_qwen3_8b")
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--split", choices=["all", "discovery", "validation", "test"], default="all")
    ap.add_argument("--limit", type=int, default=0, help="e.g., 5 for smoke test; 0 for all")
    ap.add_argument("--max-new-tokens", type=int, default=400)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    with open(args.input, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if args.split != "all":
        rows = [r for r in rows if r.get("research_split") == args.split]
    rows = [r for r in rows if r.get("question", "").strip() and
            r.get("beginner_answer", "").strip() and r.get("expert_answer", "").strip()]
    if args.limit > 0:
        rows = rows[:args.limit]
    if not rows:
        raise SystemExit("No completed answer pairs found. First populate beginner_answer and expert_answer in the CSV.")

    print(f"Loading {args.model} to evaluate {len(rows)} paired answers...", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype="auto", device_map="auto")
    model.eval()
    device = next(model.parameters()).device
    results = []
    raw_path = out / "validation_results.jsonl"
    csv_path = out / "validation_results.csv"

    for i, row in enumerate(rows, 1):
        # Randomized presentation prevents always putting the beginner answer first.
        swapped = bool(rng.getrandbits(1))
        a = row["expert_answer"] if swapped else row["beginner_answer"]
        b = row["beginner_answer"] if swapped else row["expert_answer"]
        context_present = bool((row.get("context") or "").strip())
        prompt = build_prompt(row, a, b)
        messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]
        formatted = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                             enable_thinking=False)
        inputs = tok([formatted], return_tensors="pt").to(device)
        with torch.inference_mode():
            output = model.generate(**inputs, max_new_tokens=args.max_new_tokens,
                                    do_sample=False, pad_token_id=tok.eos_token_id)
        raw = tok.decode(output[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        record = {"id": row.get("id", ""), "source": row.get("source", ""),
                  "research_split": row.get("research_split", ""), "has_context": context_present,
                  "swapped": swapped, "judge_raw": raw, "status": "parse_error"}
        try:
            j = extract_json(raw)
            if not valid(j, context_present):
                raise ValueError("Missing/invalid JSON fields or scores")
            bkey, ekey = ("B", "A") if swapped else ("A", "B")
            record.update({
                "beginner_complexity": j[f"complexity_{bkey}"],
                "expert_complexity": j[f"complexity_{ekey}"],
                "beginner_clarity": j[f"clarity_{bkey}"],
                "expert_clarity": j[f"clarity_{ekey}"],
                "beginner_groundedness": j[f"groundedness_{bkey}"],
                "expert_groundedness": j[f"groundedness_{ekey}"],
                "fact_alignment": j["fact_alignment"],
                "shared_facts_preserved": j["shared_facts_preserved"],
                "complexity_gap": j[f"complexity_{ekey}"] - j[f"complexity_{bkey}"],
                "beginner_unsupported_claims": j[f"unsupported_claims_{bkey}"],
                "expert_unsupported_claims": j[f"unsupported_claims_{ekey}"],
                "judge_explanation": j.get("explanation", ""),
                "status": "ok"
            })
            # Conservative initial filter, suitable for manual review rather than automatic truth.
            record["candidate_pass"] = (record["complexity_gap"] >= 2 and
                                        record["fact_alignment"] >= 4 and
                                        record["shared_facts_preserved"] and
                                        (not context_present or
                                         min(record["beginner_groundedness"], record["expert_groundedness"]) >= 4))
        except (ValueError, KeyError, TypeError) as exc:
            record["error"] = str(exc)
            record["candidate_pass"] = False
        results.append(record)
        # Write a checkpoint each example so interruptions do not discard completed judgments.
        with raw_path.open("a" if i > 1 else "w", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"[{i}/{len(rows)}] {record['id']} {record['status']} candidate_pass={record['candidate_pass']}", flush=True)

    cols = ["id", "source", "research_split", "has_context", "status", "candidate_pass",
            "beginner_complexity", "expert_complexity", "complexity_gap", "beginner_clarity",
            "expert_clarity", "fact_alignment", "shared_facts_preserved", "beginner_groundedness",
            "expert_groundedness", "judge_explanation"]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for r in results:
            writer.writerow({k: r.get(k, "") for k in cols})
    good = [r for r in results if r["status"] == "ok"]
    summary = {"model": args.model, "input": args.input, "evaluated": len(results),
               "valid_judgments": len(good), "candidate_pass": sum(r["candidate_pass"] for r in results),
               "context_available": sum(r["has_context"] for r in results),
               "by_source": dict(Counter(r["source"] for r in results)),
               "mean_complexity_gap": round(sum(r["complexity_gap"] for r in good) / len(good), 3) if good else None,
               "note": "LLM-judged candidate labels only; manually audit a sample and use independent judges."}
    (out / "validation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print("Saved results to", out.resolve())

if __name__ == "__main__":
    main()
