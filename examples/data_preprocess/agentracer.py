#!/usr/bin/env python3
"""
tracertraj-code -> verl RLHFDataset schema

Usage:
  python3 examples/data_preprocess/agentracer.py \
    --source /home/intern/hyomin/daia_data/agentracer/repo/data/tracertraj-code-test.parquet \
    --train-ids /home/intern/hyomin/agentracer_pipeline/data/splits/train.jsonl \
    --dev-ids   /home/intern/hyomin/agentracer_pipeline/data/splits/dev.jsonl \
    --output-dir /home/intern/hyomin/verl/data/agentracer

Design decisions (see docs):
  - output style = paper_answer:  <think>...</think> <answer>AGENT | STEP</answer>
  - ground truth code (G) is NOT shown to the model  (--with-gt to override)
  - history entries are NEVER dropped; only the invariant MetaGPT instruction
    tail is stripped from each entry's content.  Step numbers stay contiguous
    and mistake_step remains a valid 0-based index into `history`.
  - three pathological trajectories are excluded (151/125/100 entries, >16k tok)
"""

import argparse
import json
import os

import pandas as pd

# ---------------------------------------------------------------- constants

EXCLUDE_IDS = {
    "Codeforces_7286_I",  # 151 entries, ~44k tok after compaction (train)
    "Filter_70036_I",     # 100 entries, ~20k tok                  (train)
    "Filter_56113_I",     # 125 entries, ~29k tok                  (dev)
}

CANONICAL_AGENTS = [
    "Team Leader",
    "Product Manager",
    "Architect",
    "Engineer",
    "Data Analyst",
]

# Everything from this marker to the end of a MetaGPT context dump is a fixed
# system instruction, identical across agents and tasks.  Verified: only 2
# unique variants across the whole dataset.  Removing it loses no information
# and cuts median prompt length by ~39%.
BOILER_MARK = "# Response Language"
BOILER_HEAD = "# Past Experience"
BOILER_STUB = "[standard agent instructions omitted]"

ALIAS_NOTE = (
    "Note: the conversation content sometimes refers to agents by aliases. "
    "In this MetaGPT setting, Mike means Team Leader, Alice means Product Manager, "
    "Bob means Architect, and Alex means Engineer. "
    "When answering, always use the canonical name from the `Agent:` field "
    "shown at the start of each entry.\n"
)

# ---------------------------------------------------------------- helpers


def compact(content: str) -> str:
    """Strip the invariant MetaGPT instruction tail from a context-dump entry."""
    if BOILER_HEAD in content[:200] and BOILER_MARK in content:
        return content[: content.find(BOILER_MARK)] + BOILER_STUB
    return content


def build_chat_content(history) -> str:
    """Serialize history.  Entries are never removed; `step` comes from the data."""
    return "\n\n".join(
        f"Step: {e['step']}\nAgent: {e['name']}\nContent:\n{compact(e['content'])}"
        for e in history
    )


def build_prompt(question: str, chat_content: str, ground_truth: str | None) -> str:
    gt_line = f"The correct solution for the problem is:\n{ground_truth}\n\n" if ground_truth else ""
    agents = " | ".join(CANONICAL_AGENTS)
    return (
        "You are an AI assistant tasked with analyzing a multi-agent conversation "
        "history from a failed attempt at solving a programming problem.\n\n"
        f"The problem is:\n{question}\n\n"
        f"{gt_line}"
        "Identify the decisive error: the earliest step whose correction would have "
        "been sufficient to make the whole task succeed. Report which agent made it "
        "and at which step.\n\n"
        f"{ALIAS_NOTE}\n"
        "Here is the conversation:\n\n"
        f"{chat_content}\n\n"
        "Based on this conversation, determine:\n"
        "1. The agent directly responsible for the failure. If no agent makes an "
        "obvious mistake, choose the single most likely one.\n"
        "2. The step at which that agent first made the mistake. Each entry begins "
        "with an explicit `Step:` field. Use that value, not the entry's position "
        "in this prompt.\n\n"
        "First write your reasoning inside <think> and </think>.\n"
        "Then output your final answer inside <answer> and </answer>, "
        "formatted exactly as:\n\n"
        "<answer>AGENT_NAME | STEP_NUMBER</answer>\n\n"
        f"AGENT_NAME must be exactly one of: {agents}\n"
        "STEP_NUMBER must be an integer taken from a `Step:` field above.\n"
    )


def load_id_set(path: str) -> set[str]:
    """jsonl에서 문제 ID만 뽑는다. parquet은 `question_ID`, split jsonl은 `question_id`."""
    ids = set()
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            for key in ("question_ID", "question_id"):
                if key in rec:
                    ids.add(rec[key])
                    break
            else:
                raise KeyError(f"{path}: no question_ID/question_id in {list(rec)[:5]}")
    return ids


def to_verl_rows(df: pd.DataFrame, split: str, with_gt: bool) -> list[dict]:
    rows = []
    for _, r in df.iterrows():
        history = list(r["history"])
        step = int(r["mistake_step"])
        agent = str(r["mistake_agent"])

        # invariant: the label indexes the raw history (verified 127/127)
        assert 0 <= step < len(history), f"{r['question_ID']}: step {step} out of range"
        assert history[step]["name"] == agent, f"{r['question_ID']}: agent mismatch at step {step}"

        prompt_text = build_prompt(
            question=str(r["question"]),
            chat_content=build_chat_content(history),
            ground_truth=str(r["ground_truth"]) if with_gt else None,
        )

        rows.append(
            {
                "data_source": "agentracer/code",
                "prompt": [{"role": "user", "content": prompt_text}],
                "ability": "failure_attribution",
                "reward_model": {"style": "rule", "ground_truth": f"{agent} | {step}"},
                "extra_info": {
                    "split": split,
                    "question_ID": str(r["question_ID"]),
                    "mistake_agent": agent,
                    "mistake_step": step,
                    "num_steps": len(history),
                },
            }
        )
    return rows


# ---------------------------------------------------------------- main


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--train-ids", required=True)
    ap.add_argument("--dev-ids", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--with-gt", action="store_true",
                    help="include the reference solution in the prompt (w/ G setting)")
    args = ap.parse_args()

    df = pd.read_parquet(args.source)
    print(f"source: {len(df)} rows")

    train_ids = load_id_set(args.train_ids) - EXCLUDE_IDS
    dev_ids = load_id_set(args.dev_ids) - EXCLUDE_IDS
    overlap = train_ids & dev_ids
    assert not overlap, f"train/dev overlap: {overlap}"
    print(f"train ids: {len(train_ids)}   dev ids: {len(dev_ids)}   excluded: {sorted(EXCLUDE_IDS)}")

    os.makedirs(args.output_dir, exist_ok=True)

    for split, ids in (("train", train_ids), ("dev", dev_ids)):
        sub = df[df["question_ID"].isin(ids)]
        missing = ids - set(sub["question_ID"])
        assert not missing, f"{split}: ids not found in source: {missing}"

        rows = to_verl_rows(sub, split, args.with_gt)
        out = pd.DataFrame(rows)
        path = os.path.join(args.output_dir, f"{split}.parquet")
        out.to_parquet(path, index=False)
        print(f"  wrote {path}  ({len(out)} rows)")


if __name__ == "__main__":
    main()
