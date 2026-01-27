from __future__ import annotations
import random
from typing import Any, Dict, List, Optional, Sequence
import pandas as pd
from tqdm import tqdm

from syco_eval.data_utils import get_dataset, letters_for, sanitize_options
from syco_eval.enums import QFormat, Perturbation
from syco_eval.llm_utils import chat
from syco_eval.prompt_builder import (
    build_binary_messages_with_templates,
    build_binary_prompt,
    build_open_default_messages,
    build_open_messages,
    build_default_prompt,
    build_sycophancy_mc_messages,
    extract_letter,
    render_options_str,
)
from syco_eval.judge import judge


def _get_model_letter(
    messages: List[Dict[str, str]],
    model: str,
    *,
    allowed_letters: Sequence[str],
    temperature: float,
    max_tokens: int,
):
    try:
        raw = chat(messages, model=model, temperature=temperature, max_tokens=max_tokens)
        letter = extract_letter(raw, allowed_letters)
        if letter:
            return letter, raw
        raw = chat(messages, model=model, temperature=temperature)
        letter = extract_letter(raw, allowed_letters)
        return letter, raw
    except Exception as e:  # noqa: BLE001
        return None, f"ERROR: {e}"

def _sample_letters(
    messages: List[Dict[str, str]],
    model: str,
    *,
    allowed_letters: Sequence[str],
    n: int,
    temperature: float,
    max_tokens: int,
) -> List[tuple[Optional[str], str]]:
    outputs: List[tuple[Optional[str], str]] = []
    for _ in range(n):
        outputs.append(_get_model_letter(
            messages,
            model,
            allowed_letters=allowed_letters,
            temperature=temperature,
            max_tokens=max_tokens,
        ))
    return outputs

def _sample_text(
    messages: List[Dict[str, str]],
    model: str,
    *,
    n: int,
    temperature: float,
    max_tokens: int,
) -> List[str]:
    raws: List[str] = []
    for _ in range(n):
        try:
            raws.append(chat(messages, model=model, temperature=temperature, max_tokens=max_tokens))
        except Exception as e:  # noqa: BLE001
            raws.append(f"ERROR: {e}")
    return raws


def _get_default_mc_answer(
    task: str,
    q: dict,
    model: str,
    option_letters: Sequence[str],
    sol_letter: str,
    *,
    choices_override: Optional[List[str]] = None,
    n: int,
    temperature: float,
    max_tokens: int,
):
    try:
        msgs = build_default_prompt(task, q, option_letters, choices_override)
        samples = _sample_letters(
            msgs,
            model,
            allowed_letters=option_letters,
            n=n,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        defaults: List[Optional[str]] = []
        corrects: List[Optional[bool]] = []
        raws: List[str] = []
        for default, raw in samples:
            defaults.append(default)
            corrects.append((default == sol_letter) if default else None)
            raws.append(raw)
        return defaults, corrects, raws
    except Exception:
        return None, None, None


def evaluate_and_save_csv(
    task: str,
    model: str,
    out_csv: str,
    perturbation: Perturbation | str = Perturbation.SYCOPHANCY,
    *,
    max_items: Optional[int] = None,
    include_raw_cols: bool = False,
    judge_template: str = "with_Q",
    num_generations: int = 5,
    temperature: float = 0.7,
) -> pd.DataFrame:
    ds = get_dataset(task)
    pert = Perturbation(perturbation)
    if pert == Perturbation.SYCOPHANCY:
        fmt = QFormat.OPEN
    elif pert == Perturbation.FORMAT_BINARY:
        fmt = QFormat.BINARY
    else:
        fmt = QFormat.MC
    rows: List[Dict[str, Any]] = []

    def _expand(names: Sequence[str]) -> List[str]:
        return [f"{name}_{i}" for name in names for i in range(1, num_generations + 1)]

    def _set_gen_fields(row: Dict[str, Any], name: str, values: Sequence[Any]) -> None:
        for i in range(1, num_generations + 1):
            row[f"{name}_{i}"] = values[i - 1] if i - 1 < len(values) else None

    # --- column headers depending on format ---
    if fmt == QFormat.MC:
        cols = [
            "index", "question", "options", "solution",
            *_expand(["default", "default_correct", "perturbed", "perturbed_correct", "perturbation_success"]),
        ]
        if include_raw_cols:
            cols += _expand(["raw_default", "raw_perturbed"])

    elif fmt == QFormat.BINARY:
        base_cols = [
            "index", "question", "options",
            *_expand(["solution", "default", "default_correct", "perturbed", "perturbed_correct", "perturbation_success"]),
        ]
        cols = base_cols
        if include_raw_cols:
            cols = base_cols + _expand(["raw_default", "raw_perturbed"])

    else:  # OPEN
        cols = [
            "index", "question", "options", "solution_idx", "solution_text",
            *_expand(["default", "default_correct", "perturbed", "perturbed_correct", "perturbation_success"]),
        ]
        if include_raw_cols:
            cols += _expand(["raw_default", "raw_perturbed"])

    # --- loop over dataset items ---
    print(len(ds))
    for i, q in tqdm(enumerate(ds)):
        if max_items is not None and i >= max_items:
            break
        base: Dict[str, Any] = {"index": i}
        row: Dict[str, Any] = {}

        # ---------- MULTIPLE-CHOICE ----------
        if fmt == QFormat.MC:
            if task == "medqa_diag":
                q_text = q["question"].strip()
                q_text_open = q["open_ended_q"].strip()
                choices, sol_idx = sanitize_options(q["options"], (q["answer"] or "").strip().upper())
                if choices is None:
                    continue
                c_i, sol_i = choices, sol_idx
            elif task == "medxpertqa_diag":
                q_text = q["question_mc"].strip()
                q_text_open = q["question_open"].strip()
                choices, sol_idx = sanitize_options(q["options"], (q["label"] or "").strip().upper())
                if choices is None:
                    continue
                c_i, sol_i = choices, sol_idx
            else:
                continue

            c_shuf = list(c_i)
            sol_i_shuf = sol_i

            option_letters = letters_for(len(c_i))
            sol_letter = option_letters[sol_i_shuf]
            options_str = render_options_str(option_letters, c_shuf)

            base.update({"question": q_text, "options": options_str, "solution": sol_letter})

            if pert == Perturbation.FORMAT_MC:
                default_msgs = build_open_default_messages(q_text_open)
                raw_defaults = _sample_text(
                    default_msgs,
                    model,
                    n=num_generations,
                    temperature=temperature,
                    max_tokens=150,
                )
                default_corrects = [
                    judge(q_text_open, raw_default, c_i[sol_i], judge_template=judge_template)[0]
                    for raw_default in raw_defaults
                ]
                _set_gen_fields(row, "default", raw_defaults)
                _set_gen_fields(row, "default_correct", default_corrects)
                if include_raw_cols:
                    _set_gen_fields(row, "raw_default", raw_defaults)

                msgs = build_default_prompt(task, q, option_letters, choices_override=c_shuf)
                samples = _sample_letters(
                    msgs,
                    model,
                    allowed_letters=option_letters,
                    n=num_generations,
                    temperature=temperature,
                    max_tokens=30,
                )
                perturbed = [pred for pred, _ in samples]
                perturbed_corrects = [(pred == sol_letter) if pred else None for pred in perturbed]
                perturb_success = [
                    (default_corrects[i] != perturbed_corrects[i])
                    if (default_corrects[i] is not None and perturbed_corrects[i] is not None)
                    else None
                    for i in range(num_generations)
                ]
                _set_gen_fields(row, "perturbed", perturbed)
                _set_gen_fields(row, "perturbed_correct", perturbed_corrects)
                _set_gen_fields(row, "perturbation_success", perturb_success)
                if include_raw_cols:
                    _set_gen_fields(row, "raw_perturbed", [raw for _, raw in samples])
            else:
                dflt, d_ok, raw_defaults = _get_default_mc_answer(
                    task,
                    q,
                    model,
                    option_letters,
                    sol_letter,
                    choices_override=c_shuf,
                    n=num_generations,
                    temperature=temperature,
                    max_tokens=30,
                )
                if dflt is None:
                    continue
                _set_gen_fields(row, "default", dflt)
                _set_gen_fields(row, "default_correct", d_ok)
                if include_raw_cols:
                    _set_gen_fields(row, "raw_default", raw_defaults)

            if pert == Perturbation.SYCOPHANCY:
                perturbed: List[Optional[str]] = []
                perturbed_corrects: List[Optional[bool]] = []
                raw_perturbed: List[str] = []
                for i in range(num_generations):
                    if dflt[i] is None:
                        perturbed.append(None)
                        perturbed_corrects.append(None)
                        raw_perturbed.append("ERROR: missing default answer")
                        continue
                    msgs = build_sycophancy_mc_messages(task, q, dflt[i], option_letters, choices_override=c_shuf)
                    pred, raw = _get_model_letter(
                        msgs,
                        model,
                        allowed_letters=option_letters,
                        temperature=temperature,
                        max_tokens=30,
                    )
                    perturbed.append(pred)
                    perturbed_corrects.append((pred == sol_letter) if pred else None)
                    raw_perturbed.append(raw)
                perturb_success = [
                    (d_ok[i] != perturbed_corrects[i])
                    if (d_ok[i] is not None and perturbed_corrects[i] is not None)
                    else None
                    for i in range(num_generations)
                ]
                _set_gen_fields(row, "perturbed", perturbed)
                _set_gen_fields(row, "perturbed_correct", perturbed_corrects)
                _set_gen_fields(row, "perturbation_success", perturb_success)
                if include_raw_cols:
                    _set_gen_fields(row, "raw_perturbed", raw_perturbed)

        # ---------- BINARY ----------
        elif fmt == QFormat.BINARY:
            if task == "medqa_diag":
                q_text = q["open_ended_q"].strip()
                choices, sol_idx = sanitize_options(q["options"], (q["answer"] or "").strip().upper())
                if choices is None:
                    continue
                c_i, sol_i = choices, sol_idx
            elif task == "medxpertqa_diag":
                q_text = q["question_open"].strip()
                choices, sol_idx = sanitize_options(q["options"], (q["label"] or "").strip().upper())
                if choices is None:
                    continue
                c_i, sol_i = choices, sol_idx
            else:
                continue

            correct_ans = c_i[sol_i]

            mapping = {"A": "Yes", "B": "No"} if (random.random() < 0.5) else {"A": "No", "B": "Yes"}
            base.update({"question": q_text, "options": f"A. {mapping['A']}   B. {mapping['B']}"})

            # open-ended default (baseline)
            default_msgs = build_open_default_messages(q_text)
            raw_defaults = _sample_text(
                default_msgs,
                model,
                n=num_generations,
                temperature=temperature,
                max_tokens=150,
            )
            default_corrects = [
                judge(q_text, raw_default, correct_ans, judge_template=judge_template)[0]
                for raw_default in raw_defaults
            ]
            _set_gen_fields(row, "default", raw_defaults)
            _set_gen_fields(row, "default_correct", default_corrects)
            if include_raw_cols:
                _set_gen_fields(row, "raw_default", raw_defaults)

            if pert == Perturbation.FORMAT_BINARY:
                wrong_pool = [c for j, c in enumerate(c_i) if j != sol_i]
                if not wrong_pool:
                    continue
                solutions: List[Optional[str]] = []
                perturbed: List[Optional[str]] = []
                perturbed_corrects: List[Optional[bool]] = []
                raw_perturbed: List[str] = []
                for i in range(num_generations):
                    proposed_answer = random.choice(wrong_pool) if default_corrects[i] else correct_ans
                    proposed_is_correct = (proposed_answer == correct_ans)

                    msgs = build_binary_prompt(q_text, proposed_answer, mapping)
                    pred, raw = _get_model_letter(
                        msgs,
                        model,
                        allowed_letters=["A", "B"],
                        temperature=temperature,
                        max_tokens=30,
                    )
                    expected = "Yes" if proposed_is_correct else "No"
                    solutions.append(next(k for k, v in mapping.items() if v == expected))
                    perturbed.append(pred)
                    perturbed_corrects.append((mapping.get(pred or "") == expected) if pred else None)
                    raw_perturbed.append(raw)
                perturb_success = [
                    (default_corrects[i] != perturbed_corrects[i])
                    if (default_corrects[i] is not None and perturbed_corrects[i] is not None)
                    else None
                    for i in range(num_generations)
                ]
                _set_gen_fields(row, "solution", solutions)
                _set_gen_fields(row, "perturbed", perturbed)
                _set_gen_fields(row, "perturbed_correct", perturbed_corrects)
                _set_gen_fields(row, "perturbation_success", perturb_success)
                if include_raw_cols:
                    _set_gen_fields(row, "raw_perturbed", raw_perturbed)

        # ---------- OPEN-ENDED ----------
        elif fmt == QFormat.OPEN:
            if task == "medqa_diag":
                q_text = q["open_ended_q"].strip()
                choices, sol_idx = sanitize_options(q["options"], (q["answer"] or "").strip().upper())
                if choices is None:
                    continue
                c_i, sol_i = choices, sol_idx
            elif task == "medxpertqa_diag":
                q_text = q["question_open"].strip()
                choices, sol_idx = sanitize_options(q["options"], (q["label"] or "").strip().upper())
                if choices is None:
                    continue
                c_i, sol_i = choices, sol_idx
            else:
                continue

            option_letters = letters_for(len(c_i))
            sol_letter = option_letters[sol_i]
            options_str = render_options_str(option_letters, c_i)

            base.update({
                "question": q_text,
                "options": options_str,
                "solution_idx": sol_letter,
                "solution_text": c_i[sol_i].strip(),
            })

            try:
                raw_defaults = _sample_text(
                    build_open_default_messages(q_text),
                    model,
                    n=num_generations,
                    temperature=temperature,
                    max_tokens=150,
                )
            except Exception:  # noqa: BLE001
                raw_defaults = [f"ERROR: open-default failed"] * num_generations

            default_corrects = [
                judge(q_text, raw_default, c_i[sol_i], judge_template=judge_template)[0]
                for raw_default in raw_defaults
            ]
            wrong_pool = [c for c in c_i if str(c).strip() != str(c_i[sol_i]).strip()]
            display_defaults = [
                (c_i[sol_i] if default_corrects[i] else random.choice(wrong_pool))
                for i in range(num_generations)
            ]

            _set_gen_fields(row, "default", raw_defaults)
            _set_gen_fields(row, "default_correct", default_corrects)
            if include_raw_cols:
                _set_gen_fields(row, "raw_default", raw_defaults)

            perturbed: List[Optional[str]] = []
            perturbed_corrects: List[Optional[bool]] = []
            raw_perturbed: List[str] = []
            for i in range(num_generations):
                try:
                    msgs = build_open_messages(
                        q_text,
                        default_text=display_defaults[i],
                    )
                    raw = chat(msgs, model=model, temperature=temperature, max_tokens=150)
                    is_correct, _ = judge(q_text, raw, c_i[sol_i], judge_template=judge_template)
                    perturbed.append(raw)
                    perturbed_corrects.append(is_correct)
                    raw_perturbed.append(raw)
                except Exception as e:  # noqa: BLE001
                    perturbed.append(None)
                    perturbed_corrects.append(None)
                    raw_perturbed.append(f"ERROR: {e}")
            perturb_success = [
                (default_corrects[i] != perturbed_corrects[i])
                if (default_corrects[i] is not None and perturbed_corrects[i] is not None)
                else None
                for i in range(num_generations)
            ]
            _set_gen_fields(row, "perturbed", perturbed)
            _set_gen_fields(row, "perturbed_correct", perturbed_corrects)
            _set_gen_fields(row, "perturbation_success", perturb_success)
            if include_raw_cols:
                _set_gen_fields(row, "raw_perturbed", raw_perturbed)
        else:
            raise ValueError("Unknown format")
        rows.append({**base, **row})

    df = pd.DataFrame(rows)
    df = df.reindex(columns=cols)
    df.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}  (n={len(df)})")
    return df
