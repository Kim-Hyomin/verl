"""
AgenTracer multi-granular reward for verl.

    R = I_format * ( lambda * r_step + (1 - lambda) * r_agent )
    r_agent = 1[ agent == agent* ]
    r_step  = exp( -(step - step*)^2 / (2 sigma^2) )
    lambda = 0.5, sigma = 1.0                       (AgenTracer, ICLR 2026, Eq. 11-12)

Wired in via:
    reward.custom_reward_function.path=/home/intern/hyomin/verl/agentracer/reward.py
    reward.custom_reward_function.name=compute_score

No external imports: this file is loaded inside Ray workers, where
`agentracer_eval` is not on sys.path.

--- Degenerate-policy baseline (measured on tracertraj-code-test, 127 rows) ---
A constant policy that ignores the trajectory entirely and always emits
`<answer>Engineer | 4</answer>` scores **0.5997**.
(mistake_agent == Engineer in 72.4% of rows; mistake_step == 4 in 33%.)
If `critic/rewards/mean` plateaus near 0.60 while `actor/grad_norm` decays,
the policy has collapsed to that constant -- it has not learned anything.
Always read `agent_acc` / `step_exact` separately, never the total alone.
"""

import math
import re

LAMBDA = 0.5
SIGMA = 2.0

# 논문은 sigma=1.0. 그러나 학습 데이터 97개 + Qwen3-8B의 낮은 답 다양성 조건에서
# sigma=1은 그룹 내 reward 분산을 0으로 만들어 GRPO advantage가 소멸함 (실측 0/4 그룹).
# sigma=2에서 2/4 그룹 회복. verl v0.8.0의 main_ppo 경로에는 DAPO dynamic sampling
# (algorithm.filter_groups)이 구현돼 있지 않아 이 우회가 필요함.

CANONICAL_AGENTS = ("Team Leader", "Product Manager", "Architect", "Engineer", "Data Analyst")

AGENT_ALIASES = {
    "mike": "team leader",
    "team leader": "team leader",
    "teamleader": "team leader",
    "alice": "product manager",
    "product manager": "product manager",
    "productmanager": "product manager",
    "bob": "architect",
    "architect": "architect",
    "alex": "engineer",
    "engineer": "engineer",
    "engineer2": "engineer",
    "data analyst": "data analyst",
    "dataanalyst": "data analyst",
}


def normalize_agent(agent) -> str:
    key = str(agent).strip().lower().replace("_", " ")
    key = re.sub(r"\s+", " ", key)
    return AGENT_ALIASES.get(key, AGENT_ALIASES.get(key.replace(" ", ""), key))


def parse_step(x):
    m = re.search(r"-?\d+", str(x))
    return int(m.group()) if m else None


def parse_answer(text: str):
    """Return (pred_agent, pred_step, format_ok).

    The opening <think> tag is deliberately NOT required: depending on
    `enable_thinking`, the Qwen3 chat template may prefill it, in which case it
    never appears in the model's completion.  Requiring it would make
    format_ok always False -> all rewards 0 -> zero advantage -> zero gradient,
    with no error raised.  Verify against a real rollout dump before changing.
    """
    closed_think = "</think>" in text
    blocks = re.findall(r"<answer>\s*(.*?)\s*</answer>", text, re.I | re.S)
    if not blocks:
        return "", None, False

    body = blocks[-1]  # the model may rehearse an answer inside <think>
    if "|" not in body:
        return body.strip(), None, False

    agent_text, step_text = body.rsplit("|", 1)
    step = parse_step(step_text)
    return agent_text.strip(), step, bool(closed_think and step is not None)


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    info = extra_info or {}
    gold_agent = info.get("mistake_agent")
    gold_step = info.get("mistake_step")

    # fall back to parsing "Agent | Step" out of ground_truth if extra_info is absent
    if gold_agent is None or gold_step is None:
        gt = str(ground_truth)
        if "|" in gt:
            a, s = gt.rsplit("|", 1)
            gold_agent, gold_step = a.strip(), parse_step(s)

    gold_step = parse_step(gold_step) if gold_step is not None else None

    pred_agent, pred_step, format_ok = parse_answer(str(solution_str))

    r_format = 1.0 if format_ok else 0.0
    r_agent = 1.0 if normalize_agent(pred_agent) == normalize_agent(gold_agent) else 0.0
    if pred_step is None or gold_step is None:
        r_step = 0.0
    else:
        r_step = math.exp(-((pred_step - gold_step) ** 2) / (2.0 * SIGMA**2))

    score = r_format * (LAMBDA * r_step + (1.0 - LAMBDA) * r_agent)

    return {
        "score": score,
        "format": r_format,
        "agent_acc": r_agent,
        "step_exact": 1.0 if (pred_step is not None and pred_step == gold_step) else 0.0,
        "step_gauss": r_step,
    }
