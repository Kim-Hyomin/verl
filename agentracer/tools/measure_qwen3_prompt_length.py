from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from transformers import AutoTokenizer


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []

    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{path}의 {line_number}번째 줄을 JSON으로 읽지 못했습니다."
                ) from exc

    return rows


def normalize_messages(value: Any) -> list[dict[str, str]] | None:
    """
    다음과 같은 chat message 형식을 인식한다.

    [
        {"role": "system", "content": "..."},
        {"role": "user", "content": "..."}
    ]
    """
    if isinstance(value, np.ndarray):
        value = value.tolist()

    if isinstance(value, str):
        stripped = value.strip()

        if stripped.startswith("["):
            try:
                parsed = json.loads(stripped)
            except json.JSONDecodeError:
                return None

            if isinstance(parsed, list):
                value = parsed
            else:
                return None
        else:
            return None

    if not isinstance(value, list):
        return None

    messages: list[dict[str, str]] = []

    for item in value:
        if not isinstance(item, dict):
            try:
                item = dict(item)
            except (TypeError, ValueError):
                return None

        if "role" not in item or "content" not in item:
            return None

        content = item.get("content", "")

        messages.append(
            {
                "role": str(item.get("role", "user")),
                "content": "" if content is None else str(content),
            }
        )

    return messages


def find_prompt_value(
    row: dict[str, Any],
    prompt_column: str | None,
) -> tuple[Any, str]:
    """
    명시된 prompt 컬럼 또는 흔히 사용되는 컬럼을 찾는다.
    """
    if prompt_column:
        if prompt_column not in row:
            raise KeyError(
                f"지정한 prompt column '{prompt_column}'이 없습니다. "
                f"현재 keys={list(row.keys())}"
            )

        return row[prompt_column], prompt_column

    candidates = [
        "prompt",
        "messages",
        "message",
        "input",
        "query",
        "instruction",
    ]

    for key in candidates:
        if key in row:
            return row[key], key

    raise KeyError(
        "prompt로 사용할 컬럼을 자동으로 찾지 못했습니다. "
        f"현재 keys={list(row.keys())}\n"
        "--prompt-column 옵션으로 실제 컬럼을 지정하세요."
    )


def make_prompt_text(
    value: Any,
    tokenizer,
    add_generation_prompt: bool,
) -> tuple[str, str]:
    messages = normalize_messages(value)

    if messages is not None:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
        return text, "chat_template"

    if isinstance(value, dict):
        # dict 자체가 들어 있는 경우 임시로 JSON 문자열화한다.
        # 실제 학습 prompt 생성 코드와 다를 수 있으므로 결과에서 확인 필요.
        text = json.dumps(value, ensure_ascii=False)
        return text, "json_dict"

    if isinstance(value, list):
        text = json.dumps(value, ensure_ascii=False)
        return text, "json_list"

    return str(value), "plain_text"


def print_stats(
    result: pd.DataFrame,
    max_prompt_length: int,
) -> None:
    lengths = result["token_length"].to_numpy()
    chars_per_token = result["chars_per_token"].dropna().to_numpy()

    print("\n" + "=" * 72)
    print("Token length")
    print("=" * 72)

    for name, value in [
        ("min", np.min(lengths)),
        ("mean", np.mean(lengths)),
        ("median", np.median(lengths)),
        ("p90", np.percentile(lengths, 90)),
        ("p95", np.percentile(lengths, 95)),
        ("p99", np.percentile(lengths, 99)),
        ("max", np.max(lengths)),
    ]:
        print(f"{name:>8}: {value:,.2f}")

    print("\n" + "=" * 72)
    print("Actual chars per token")
    print("=" * 72)

    for name, value in [
        ("mean", np.mean(chars_per_token)),
        ("median", np.median(chars_per_token)),
        ("p10", np.percentile(chars_per_token, 10)),
        ("p90", np.percentile(chars_per_token, 90)),
    ]:
        print(f"{name:>8}: {value:,.4f}")

    exceeded = result["exceeds_limit"].sum()
    total = len(result)

    print("\n" + "=" * 72)
    print(f"max_prompt_length = {max_prompt_length:,}")
    print("=" * 72)
    print(f"초과 샘플: {exceeded}/{total} ({exceeded / total * 100:.2f}%)")

    thresholds = [8192, 12288, 16384, 20480, 24576, 28672, 32768]

    print("\n길이 구간별 누적 초과율")

    for threshold in thresholds:
        count = int((result["token_length"] > threshold).sum())
        print(
            f"  > {threshold:>5,}: "
            f"{count:>3}/{total} ({count / total * 100:>6.2f}%)"
        )

    print("\n가장 긴 샘플 10개")

    columns = [
        "row_index",
        "source_column",
        "prompt_type",
        "char_length",
        "token_length",
        "chars_per_token",
        "overflow_tokens",
    ]

    print(
        result.nlargest(10, "token_length")[columns].to_string(index=False)
    )


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--data", required=True)
    parser.add_argument(
        "--model",
        default="/home/intern/hyomin/models/Qwen3-8B",
    )
    parser.add_argument("--prompt-column", default=None)
    parser.add_argument("--max-prompt-length", type=int, default=24576)
    parser.add_argument("--add-generation-prompt", action="store_true")
    parser.add_argument("--output", default=None)

    args = parser.parse_args()

    data_path = Path(args.data)

    if not data_path.exists():
        raise FileNotFoundError(data_path)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=True,
        local_files_only=True,
    )

    suffix = data_path.suffix.lower()

    if suffix == ".jsonl":
        rows = read_jsonl(data_path)
    elif suffix == ".parquet":
        rows = pd.read_parquet(data_path).to_dict(orient="records")
    else:
        raise ValueError("현재 .jsonl과 .parquet만 지원합니다.")

    records = []

    for row_index, row in enumerate(rows):
        prompt_value, source_column = find_prompt_value(
            row=row,
            prompt_column=args.prompt_column,
        )

        prompt_text, prompt_type = make_prompt_text(
            value=prompt_value,
            tokenizer=tokenizer,
            add_generation_prompt=args.add_generation_prompt,
        )

        token_ids = tokenizer.encode(
            prompt_text,
            add_special_tokens=False,
        )

        char_length = len(prompt_text)
        token_length = len(token_ids)

        records.append(
            {
                "row_index": row_index,
                "source_column": source_column,
                "prompt_type": prompt_type,
                "char_length": char_length,
                "token_length": token_length,
                "chars_per_token": (
                    char_length / token_length
                    if token_length
                    else np.nan
                ),
                "chars_div_3_0": char_length / 3.0,
                "chars_div_3_6": char_length / 3.6,
                "error_chars_div_3_0": (
                    char_length / 3.0 - token_length
                ),
                "error_chars_div_3_6": (
                    char_length / 3.6 - token_length
                ),
                "exceeds_limit": token_length > args.max_prompt_length,
                "overflow_tokens": max(
                    token_length - args.max_prompt_length,
                    0,
                ),
            }
        )

    result = pd.DataFrame(records)

    print("=" * 72)
    print("Measurement configuration")
    print("=" * 72)
    print("data:", data_path)
    print("rows:", len(result))
    print("model:", args.model)
    print("tokenizer:", tokenizer.__class__.__name__)
    print("add_generation_prompt:", args.add_generation_prompt)
    print(
        "source columns:",
        result["source_column"].value_counts().to_dict(),
    )
    print(
        "prompt types:",
        result["prompt_type"].value_counts().to_dict(),
    )

    print_stats(
        result=result,
        max_prompt_length=args.max_prompt_length,
    )

    output_path = (
        Path(args.output)
        if args.output
        else Path(f"{data_path.stem}_qwen3_token_lengths.csv")
    )

    result.to_csv(output_path, index=False)

    print("\n상세 결과 저장:", output_path.resolve())


if __name__ == "__main__":
    main()
