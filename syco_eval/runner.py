from __future__ import annotations
import random
from typing import Any, Dict, List, Optional, Sequence
import pandas as pd
from tqdm import tqdm

from data_utils import get_dataset, letters_for, sanitize_options
from enums import QFormat, Template
from llm_utils import chat
from prompt_builder import (
    build_binary_messages_with_templates,
    build_binary_prompt,
    build_open_default_messages,
    build_open_messages,
    build_default_prompt,
    build_sycophancy_mc_messages,
    extract_letter,
    render_options_str,
)
from judge import judge


def _get_model_letter(messages: List[Dict[str, str]], model: str, *, allowed_letters: Sequence[str]):
    try:
        raw = chat(messages, model=model, temperature=0.0, max_tokens=30)
        letter = extract_letter(raw, allowed_letters)
        if letter:
            return letter, raw
        raw = chat(messages, model=model, temperature=0.0)
        letter = extract_letter(raw, allowed_letters)
        return letter, raw
    except Exception as e:  # noqa: BLE001
        return None, f"ERROR: {e}"


def _get_default_mc_answer(
    task: str,
    q: dict,
    model: str,
    option_letters: Sequence[str],
    sol_letter: str,
    *,
    choices_override: Optional[List[str]] = None
):
    try:
        msgs = build_default_prompt(task, q, option_letters, choices_override)
        raw = chat(msgs, model=model, temperature=0.0, max_tokens=30)
        default = extract_letter(raw, allowed_letters=option_letters)
        if not default:
            raw = chat(msgs, model=model, temperature=0.0)
            default = extract_letter(raw, allowed_letters=option_letters)
        if not default:
            return None, None, None
        is_correct = (default == sol_letter)
        alt = random.choice([L for L in option_letters if L != default]) if is_correct else sol_letter
        return default, alt, is_correct
    except Exception:
        return None, None, None


def evaluate_and_save_csv(
    task: str,
    model: str,
    format: QFormat | str,
    out_csv: str,
    *,
    shuffle_seed: int = 0,
    max_items: Optional[int] = None,
    include_raw_cols: bool = False
) -> pd.DataFrame:
    ds = get_dataset(task)
    fmt = QFormat(format)

    rng = random.Random(shuffle_seed)
    rows: List[Dict[str, Any]] = []

    # --- column headers depending on format ---
    if fmt == QFormat.MC:
        cols = [
            "index", "question", "options", "solution",
            "default", "alternative", "default_correct",
            "pred_a", "correct_a",
            "pred_b", "correct_b",
            "pred_c", "correct_c",
            "pred_d", "correct_d",
        ]
        if include_raw_cols:
            cols += ["raw_a", "raw_b", "raw_c", "raw_d"]

    elif fmt == QFormat.BINARY:
        base_cols = [
            "index", "question", "options",
            "default_true", "solution_true", "default_correct_true",
            "default_false", "solution_false", "default_correct_false",
        ]
        var_cols = []
        for t in ["a", "b", "c", "d"]:
            var_cols += [f"pred_{t}_true", f"correct_{t}_true", *( [f"raw_{t}_true"] if include_raw_cols else [] )]
            var_cols += [f"pred_{t}_false", f"correct_{t}_false", *( [f"raw_{t}_false"] if include_raw_cols else [] )]
        cols = base_cols + var_cols

    else:  # OPEN
        cols = [
            "index", "question", "options", "solution_idx", "solution_text",
            "default", "alternative", "default_correct",
            "correct_a", *( ["raw_a"] if include_raw_cols else [] ),
            "correct_b", *( ["raw_b"] if include_raw_cols else [] ),
            "correct_c", *( ["raw_c"] if include_raw_cols else [] ),
            "correct_d", *( ["raw_d"] if include_raw_cols else [] ),
        ]

    # --- loop over dataset items ---
    print(len(ds))
    for i, q in tqdm(enumerate(ds)):
        if max_items is not None and i >= max_items:
            break
        base: Dict[str, Any] = {"index": i}
        row: Dict[str, Any] = {}

        # ---------- MULTIPLE-CHOICE ----------
        if fmt == QFormat.MC:
            if task in {"mmlu_elem_math", "mmlu_high_school_math", "mmlu_college_math"}:
                q_text = q["question"].strip()
                c_i = list(q["choices"])
                sol_i = int(q["answer"]) if isinstance(q["answer"], int) else q["answer"]
            elif task == "ethics_commonsense":
                q_text = q["input"].strip()
                c_i = ["No", "Yes"]
                sol_i = int(q["label"])
            elif task == "truthfulqa_mc1":
                q_text = q["question"].strip()
                c_i = list(q["mc1_targets"]["choices"])
                try:
                    sol_i = q["mc1_targets"]["labels"].index(1)
                except ValueError:
                    continue
            elif task == "truthfulqa_binary":
                q_text = q["Question"].strip()
                c_i = [q["Best Answer"], q["Best Incorrect Answer"]]
                sol_i = 0
            elif task == "medqa_4":
                q_text = q["question"].strip()
                raw = q["options"]
                c_i = [raw[k] for k in sorted(raw.keys())]
                sol_i = ord(q["answer_idx"]) - ord("A")
            elif task == "medqa_diag":
                q_text = q["question"].strip()
                choices, sol_idx = sanitize_options(q["options"], (q["answer"] or "").strip().upper())
                if choices is None:
                    continue
                c_i, sol_i = choices, sol_idx
            elif task == "medxpertqa_diag":
                q_text = q["question_mc"].strip()
                choices, sol_idx = sanitize_options(q["options"], (q["label"] or "").strip().upper())
                if choices is None:
                    continue
                c_i, sol_i = choices, sol_idx
            else:
                continue

            # shuffle if needed
            do_shuffle = task in {"truthfulqa_mc1", "truthfulqa_binary"}
            perm = list(range(len(c_i)))
            if do_shuffle:
                local_rng = random.Random(rng.random() + i)
                local_rng.shuffle(perm)
            c_shuf = [c_i[j] for j in perm]
            sol_i_shuf = perm.index(sol_i)

            option_letters = letters_for(len(c_i))
            sol_letter = option_letters[sol_i_shuf]
            options_str = render_options_str(option_letters, c_shuf)

            base.update({"question": q_text, "options": options_str, "solution": sol_letter})

            dflt, alt, d_ok = _get_default_mc_answer(task, q, model, option_letters, sol_letter, choices_override=c_shuf)
            if dflt is None or alt is None:
                continue
            row.update({"default": dflt, "alternative": alt, "default_correct": d_ok})

            for templ in [Template.A, Template.B, Template.C, Template.D]:
                msgs = build_sycophancy_mc_messages(task, q, templ, dflt, alt, option_letters, choices_override=c_shuf)
                pred, raw = _get_model_letter(msgs, model, allowed_letters=option_letters)
                row[f"pred_{templ.value}"] = pred
                row[f"correct_{templ.value}"] = (pred == sol_letter) if pred else None
                if include_raw_cols:
                    row[f"raw_{templ.value}"] = raw

        # ---------- BINARY ----------
        elif fmt == QFormat.BINARY:
            if task == "truthfulqa_binary":
                q_text = q["Question"].strip()
                correct_ans = q["Best Answer"]
                wrong_ans = q["Best Incorrect Answer"]
            elif task in {
                "mmlu_elem_math", "mmlu_high_school_math", "mmlu_college_math",
                "truthfulqa_mc1", "medqa_4", "medqa_diag", "medxpertqa_diag", "ethics_commonsense"
            }:
                if task in {"mmlu_elem_math", "mmlu_high_school_math", "mmlu_college_math"}:
                    q_text = q["question"].strip()
                    c_i = list(q["choices"])
                    sol_i = int(q["answer"]) if isinstance(q["answer"], int) else q["answer"]
                elif task == "ethics_commonsense":
                    q_text = q["input"].strip()
                    c_i = ["No", "Yes"]
                    sol_i = int(q["label"])
                elif task == "truthfulqa_mc1":
                    q_text = q["question"].strip()
                    c_i = list(q["mc1_targets"]["choices"])
                    try:
                        sol_i = q["mc1_targets"]["labels"].index(1)
                    except ValueError:
                        continue
                elif task == "medqa_4":
                    q_text = q["question"].strip()
                    raw = q["options"]
                    c_i = [raw[k] for k in sorted(raw.keys())]
                    sol_i = ord(q["answer_idx"]) - ord("A")
                elif task == "medqa_diag":
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

                correct_ans = c_i[sol_i]
                wrong_pool = [c for j, c in enumerate(c_i) if j != sol_i]
                if not wrong_pool:
                    continue
                wrong_ans = random.choice(wrong_pool)
            else:
                continue

            mapping = {"A": "Yes", "B": "No"} if (random.random() < 0.5) else {"A": "No", "B": "Yes"}
            base.update({"question": q_text, "options": f"A. {mapping['A']}   B. {mapping['B']}"})

            # TRUE
            msgs_true = build_binary_prompt(q_text, correct_ans, mapping)
            d_true_letter, raw_true = _get_model_letter(msgs_true, model, allowed_letters=["A", "B"])
            default_correct_true = (mapping.get(d_true_letter or "") == "Yes")
            row.update({
                "default_true": d_true_letter,
                "solution_true": next(k for k, v in mapping.items() if v == "Yes"),
                "default_correct_true": default_correct_true,
            })
            if include_raw_cols:
                row["raw_default_true"] = raw_true

            alt_true = "A" if d_true_letter == "B" else "B"
            for templ in [Template.A, Template.B, Template.C, Template.D]:
                msgs = build_binary_messages_with_templates(
                    q_text, correct_ans, mapping,
                    default_letter=d_true_letter,
                    alternative_letter=alt_true,
                    template=templ
                )
                pred, raw = _get_model_letter(msgs, model, allowed_letters=["A", "B"])
                row[f"pred_{templ.value}_true"] = pred
                row[f"correct_{templ.value}_true"] = (mapping.get(pred or "") == "Yes") if pred else None
                if include_raw_cols:
                    row[f"raw_{templ.value}_true"] = raw

            # FALSE
            msgs_false = build_binary_prompt(q_text, wrong_ans, mapping)
            d_false_letter, raw_false = _get_model_letter(msgs_false, model, allowed_letters=["A", "B"])
            default_correct_false = (mapping.get(d_false_letter or "") == "No")
            row.update({
                "default_false": d_false_letter,
                "solution_false": next(k for k, v in mapping.items() if v == "No"),
                "default_correct_false": default_correct_false,
            })
            if include_raw_cols:
                row["raw_default_false"] = raw_false

            alt_false = "A" if d_false_letter == "B" else "B"
            for templ in [Template.A, Template.B, Template.C, Template.D]:
                msgs = build_binary_messages_with_templates(
                    q_text, wrong_ans, mapping,
                    default_letter=d_false_letter,
                    alternative_letter=alt_false,
                    template=templ
                )
                pred, raw = _get_model_letter(msgs, model, allowed_letters=["A", "B"])
                row[f"pred_{templ.value}_false"] = pred
                row[f"correct_{templ.value}_false"] = (mapping.get(pred or "") == "No") if pred else None
                if include_raw_cols:
                    row[f"raw_{templ.value}_false"] = raw

        # ---------- OPEN-ENDED ----------
        elif fmt == QFormat.OPEN:
            if task in {"mmlu_elem_math", "mmlu_high_school_math", "mmlu_college_math"}:
                q_text = q["question"].strip()
                c_i = list(q["choices"])
                sol_i = int(q["answer"]) if isinstance(q["answer"], int) else q["answer"]
            elif task == "ethics_commonsense":
                q_text = q["input"].strip()
                c_i = ["No", "Yes"]
                sol_i = int(q["label"])
            elif task == "truthfulqa_mc1":
                q_text = q["question"].strip()
                c_i = list(q["mc1_targets"]["choices"])
                try:
                    sol_i = q["mc1_targets"]["labels"].index(1)
                except ValueError:
                    continue
            elif task == "truthfulqa_binary":
                q_text = q["Question"].strip()
                c_i = [q["Best Answer"], q["Best Incorrect Answer"]]
                sol_i = 0
            elif task == "medqa_4":
                q_text = q["question"].strip()
                raw = q["options"]
                c_i = [raw[k] for k in sorted(raw.keys())]
                sol_i = ord(q["answer_idx"]) - ord("A")
            elif task == "medqa_diag":
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
                raw_default = chat(build_open_default_messages(q_text), model=model, temperature=0.0, max_tokens=150)
            except Exception as e:  # noqa: BLE001
                raw_default = f"ERROR: {e}"

            default_correct, _ = judge(q_text, raw_default, c_i[sol_i])
            wrong_pool = [c for c in c_i if str(c).strip() != str(c_i[sol_i]).strip()]
            alternative_text = random.choice(wrong_pool) if default_correct and wrong_pool else c_i[sol_i]

            # NEW: pick a display_default_text that is always one of the options
            display_default_text = c_i[sol_i] if default_correct else random.choice(wrong_pool) 
            
            row.update({"default": raw_default, "alternative": alternative_text, "default_correct": default_correct})

            for templ in [Template.A, Template.B, Template.C, Template.D]:
                try:
                    msgs = build_open_messages(
                        q_text,
                        templ,
                        default_text=display_default_text,
                        alternative_text=alternative_text,
                    )
                    raw = chat(msgs, model=model, temperature=0.0, max_tokens=150)
                    is_correct, _ = judge(q_text, raw, c_i[sol_i])
                    row[f"correct_{templ.value}"] = is_correct
                    if include_raw_cols:
                        row[f"raw_{templ.value}"] = raw
                except Exception as e:  # noqa: BLE001
                    row[f"correct_{templ.value}"] = None
                    if include_raw_cols:
                        row[f"raw_{templ.value}"] = f"ERROR: {e}"
        else:
            raise ValueError("Unknown format")
        rows.append({**base, **row})

    df = pd.DataFrame(rows)
    df = df.reindex(columns=cols)
    df.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}  (n={len(df)})")
    return df
