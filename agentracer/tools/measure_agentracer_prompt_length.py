from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from transformers import AutoTokenizer


MODEL_PATH = "/home/intern/hyomin/models/Qwen3-8B"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{path}의 {line_number}번째 줄을 읽지 못했습니다."
                ) from exc

    return rows


def safe_text(value: Any) -> str:
    if value is None:
        return ""

    if isinstance(value, str):
        return value

    return json.dumps(
        value,
        ensure_ascii=False,
        indent=2,
        default=str,
    )


def get_first_present(
    item: dict[str, Any],
    keys: list[str],
    default: Any = "",
) -> Any:
    for key in keys:
        if key in item and item[key] is not None:
            return item[key]

    return default


def format_history_entry(
    entry: Any,
    position: int,
) -> str:
    """
    history의 각 entry를 다음 형태로 변환한다.

    Step: ...
    Agent: ...
    Content: ...

    실제 key 이름이 조금 달라도 대응하도록 여러 후보 key를 둔다.
    """
    if not isinstance(entry, dict):
        return (
            f"Step: {position}\n"
            f"Agent: Unknown\n"
            f"Content: {safe_text(entry)}"
        )

    step = get_first_present(
        entry,
        [
            "step",
            "step_number",
            "step_id",
            "index",
            "turn",
            "turn_id",
        ],
        default=position,
    )

    agent = get_first_present(
        entry,
        [
            "agent",
            "agent_name",
            "name",
            "role",
            "speaker",
            "sender",
        ],
        default="Unknown",
    )

    content = get_first_present(
        entry,
        [
            "content",
            "message",
            "text",
            "action",
            "response",
            "output",
            "thought",
        ],
        default=None,
    )

    # 명시적인 content key를 찾지 못한 경우, step과 agent를 제외한
    # 나머지 필드를 모두 보존한다.
    if content is None:
        excluded_keys = {
            "step",
            "step_number",
            "step_id",
            "index",
            "turn",
            "turn_id",
            "agent",
            "agent_name",
            "name",
            "role",
            "speaker",
            "sender",
        }

        remaining = {
            key: value
            for key, value in entry.items()
            if key not in excluded_keys
        }

        content = remaining if remaining else entry

    return (
        f"Step: {safe_text(step)}\n"
        f"Agent: {safe_text(agent)}\n"
        f"Content: {safe_text(content)}"
    )


def format_history(history: Any) -> str:
    """
    history 전체를 학습·평가 prompt에 넣을 평문으로 변환한다.
    """
    if history is None:
        return ""

    # history가 JSON 문자열로 저장된 경우 처리
    if isinstance(history, str):
        stripped = history.strip()

        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            return stripped

        history = parsed

    if isinstance(history, np.ndarray):
        history = history.tolist()

    if isinstance(history, list):
        formatted_entries = [
            format_history_entry(entry, position=index)
            for index, entry in enumerate(history, start=1)
        ]

        return "\n\n".join(formatted_entries)

    if isinstance(history, dict):
        # 일부 데이터에서 history 안에 다시 list가 들어 있을 가능성 처리
        for key in ["history", "messages", "conversation", "trajectory", "steps"]:
            if key in history and isinstance(history[key], list):
                return format_history(history[key])

        return format_history_entry(history, position=1)

    return safe_text(history)


def build_prompt(
    row: dict[str, Any],
    include_ground_truth: bool,
) -> tuple[str, str]:
    problem = safe_text(row.get("question", ""))

    history = row.get("history", [])
    chat_content = format_history(history)

    if include_ground_truth:
        ground_truth = safe_text(row.get("ground_truth", ""))

        ground_truth_line = (
            f"The correct answer to the problem is: {ground_truth}\n"
        )
        mode = "with_ground_truth"
    else:
        ground_truth_line = ""
        mode = "without_ground_truth"

    prompt = (
        "You are an AI assistant tasked with analyzing a multi-agent "
        "conversation history when solving a real world problem. "
        f"The problem is:  {problem}\n"
        f"{ground_truth_line}"
        "Identify which agent made an error, at which step, and explain "
        "the reason for the error. "
        "Note: The content may refer to agents by aliases. In this MetaGPT "
        "setting, Mike means Team Leader, Alice means Product Manager, "
        "Bob means Architect, and Alex means Engineer. When answering "
        "Agent Name, use the canonical Agent field shown at the start of "
        "each entry.\n"
        "Here's the conversation:\n\n"
        + chat_content
        + "\n\nBased on this conversation, please predict the following:\n"
        "1. The name of the agent who made a mistake that should be directly "
        "responsible for the wrong solution to the real world problem. "
        "If there are no agents that make obvious mistakes, decide one "
        "single agent in your mind. Directly output the name of the Expert.\n"
        "2. In which step the mistake agent first made mistake. Each "
        "conversation entry starts with explicit Step and Agent fields. "
        "Please use the explicit Step value shown in the conversation, not "
        "the line number or the entry position in the prompt. Please "
        "determine the step number where the first mistake occurred.\n"
        "3. The reason for your prediction.\n"
        "Please first write your reasoning inside <think> and </think>. "
        "Then answer in the format:\n"
        "Agent Name: (Your prediction)\n"
        "Step Number: (Your prediction)\n"
        "Reason for Mistake: (Your reason)\n"
    )

    return prompt, mode


def tokenize_prompt(
    prompt: str,
    tokenizer,
    apply_chat_template: bool,
) -> tuple[int, str]:
    """
    실제 rollout이 messages 형태라면 chat template를 적용한다.

    messages = [{"role": "user", "content": prompt}]
    """
    if apply_chat_template:
        messages = [
            {
                "role": "user",
                "content": prompt,
            }
        ]

        token_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
        )

        return len(token_ids), "qwen_chat_template"

    token_ids = tokenizer.encode(
        prompt,
        add_special_tokens=False,
    )

    return len(token_ids), "raw_prompt"


def analyze_file(
    data_path: Path,
    tokenizer,
    include_ground_truth: bool,
    apply_chat_template: bool,
    max_prompt_length: int,
) -> tuple[pd.DataFrame, list[str]]:
    rows = read_jsonl(data_path)

    records: list[dict[str, Any]] = []
    prompts: list[str] = []

    for row_index, row in enumerate(rows):
        prompt, mode = build_prompt(
            row=row,
            include_ground_truth=include_ground_truth,
        )

        token_length, tokenization_mode = tokenize_prompt(
            prompt=prompt,
            tokenizer=tokenizer,
            apply_chat_template=apply_chat_template,
        )

        char_length = len(prompt)
        chars_per_token = (
            char_length / token_length
            if token_length > 0
            else np.nan
        )

        records.append(
            {
                "row_index": row_index,
                "question_id": row.get("question_id", row_index),
                "split_file": data_path.name,
                "prompt_mode": mode,
                "tokenization_mode": tokenization_mode,
                "history_entries": (
                    len(row.get("history", []))
                    if isinstance(row.get("history"), list)
                    else np.nan
                ),
                "char_length": char_length,
                "token_length": token_length,
                "chars_per_token": chars_per_token,
                "estimate_chars_div_3_0": char_length / 3.0,
                "estimate_chars_div_3_6": char_length / 3.6,
                "error_div_3_0": char_length / 3.0 - token_length,
                "error_div_3_6": char_length / 3.6 - token_length,
                "exceeds_24576": token_length > 24576,
                "exceeds_limit": token_length > max_prompt_length,
                "overflow_tokens": max(
                    token_length - max_prompt_length,
                    0,
                ),
            }
        )

        prompts.append(prompt)

    return pd.DataFrame(records), prompts


def print_statistics(
    df: pd.DataFrame,
    max_prompt_length: int,
) -> None:
    tokens = df["token_length"]
    ratios = df["chars_per_token"]

    print("\n" + "=" * 72)
    print("Token length statistics")
    print("=" * 72)

    statistics = {
        "min": tokens.min(),
        "mean": tokens.mean(),
        "median": tokens.median(),
        "p90": tokens.quantile(0.90),
        "p95": tokens.quantile(0.95),
        "p99": tokens.quantile(0.99),
        "max": tokens.max(),
    }

    for name, value in statistics.items():
        print(f"{name:>8}: {value:,.2f}")

    print("\n" + "=" * 72)
    print("Chars per token")
    print("=" * 72)
    print(f"{'mean':>8}: {ratios.mean():,.4f}")
    print(f"{'median':>8}: {ratios.median():,.4f}")
    print(f"{'p10':>8}: {ratios.quantile(0.10):,.4f}")
    print(f"{'p90':>8}: {ratios.quantile(0.90):,.4f}")

    actual = df["token_length"]

    for divisor in [3.0, 3.6]:
        estimate = df["char_length"] / divisor
        error = estimate - actual
        relative_error = error.abs() / actual * 100

        print(f"\nchars / {divisor:.1f}")
        print(f"  MAE: {error.abs().mean():,.2f} tokens")
        print(f"  mean relative error: {relative_error.mean():,.2f}%")
        print(f"  mean bias: {error.mean():,.2f} tokens")

        if error.mean() > 0:
            print("  direction: 토큰 수를 평균적으로 과대 추정")
        else:
            print("  direction: 토큰 수를 평균적으로 과소 추정")

    print("\n" + "=" * 72)
    print("Threshold analysis")
    print("=" * 72)

    thresholds = [
        8192,
        12288,
        16384,
        20480,
        24576,
        28672,
        32768,
    ]

    total = len(df)

    for threshold in thresholds:
        count = int((tokens > threshold).sum())
        ratio = count / total * 100

        print(
            f"> {threshold:>6,}: "
            f"{count:>3}/{total} ({ratio:>6.2f}%)"
        )

    exceeded = int((tokens > max_prompt_length).sum())

    print(
        f"\n설정값 {max_prompt_length:,} 초과: "
        f"{exceeded}/{total} "
        f"({exceeded / total * 100:.2f}%)"
    )

    print("\n가장 긴 샘플 10개")

    columns = [
        "question_id",
        "split_file",
        "history_entries",
        "char_length",
        "token_length",
        "chars_per_token",
        "overflow_tokens",
    ]

    print(
        df.nlargest(10, "token_length")[columns].to_string(index=False)
    )


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data",
        nargs="+",
        required=True,
        help="하나 이상의 JSONL 경로",
    )
    parser.add_argument(
        "--model",
        default=MODEL_PATH,
    )
    parser.add_argument(
        "--max-prompt-length",
        type=int,
        default=24576,
    )
    parser.add_argument(
        "--with-ground-truth",
        action="store_true",
    )
    parser.add_argument(
        "--raw-prompt",
        action="store_true",
        help="Qwen chat template를 적용하지 않고 prompt 문자열만 측정",
    )
    parser.add_argument(
        "--output",
        default="agentracer_qwen3_token_lengths.csv",
    )
    parser.add_argument(
        "--save-longest-prompt",
        default="longest_prompt.txt",
    )

    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=True,
        local_files_only=True,
    )

    print("=" * 72)
    print("Tokenizer")
    print("=" * 72)
    print("model path:", args.model)
    print("class:", tokenizer.__class__.__name__)
    print("vocab size:", len(tokenizer))
    print("model max length:", tokenizer.model_max_length)
    print("chat template available:", tokenizer.chat_template is not None)

    all_results: list[pd.DataFrame] = []
    all_prompts: list[str] = []

    for raw_path in args.data:
        data_path = Path(raw_path)

        if not data_path.exists():
            raise FileNotFoundError(data_path)

        result, prompts = analyze_file(
            data_path=data_path,
            tokenizer=tokenizer,
            include_ground_truth=args.with_ground_truth,
            apply_chat_template=not args.raw_prompt,
            max_prompt_length=args.max_prompt_length,
        )

        all_results.append(result)
        all_prompts.extend(prompts)

    df = pd.concat(all_results, ignore_index=True)

    print("\n" + "=" * 72)
    print("Measurement configuration")
    print("=" * 72)
    print("rows:", len(df))
    print("files:", args.data)
    print("with ground truth:", args.with_ground_truth)
    print("raw prompt:", args.raw_prompt)
    print("tokenization mode:", df["tokenization_mode"].unique().tolist())

    print_statistics(
        df=df,
        max_prompt_length=args.max_prompt_length,
    )

    output_path = Path(args.output)
    df.to_csv(output_path, index=False)

    longest_index = int(df["token_length"].idxmax())
    longest_prompt = all_prompts[longest_index]

    longest_prompt_path = Path(args.save_longest_prompt)
    longest_prompt_path.write_text(
        longest_prompt,
        encoding="utf-8",
    )

    print("\n결과 CSV:", output_path.resolve())
    print("최장 prompt:", longest_prompt_path.resolve())


if __name__ == "__main__":
    main()
