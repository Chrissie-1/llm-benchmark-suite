"""benchmark: a small, reproducible evaluation harness.

Evaluates three causal LMs on MMLU (5-shot, multiple choice) and
GSM8K (8-shot, chain-of-thought), recording accuracy, throughput and
peak GPU memory. Batch size is 1 throughout so results are comparable
and fit in 8 GB of VRAM.
"""

import argparse
import gc
import json
import re
import time
from pathlib import Path

import pandas as pd
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

MODELS = [
    ("Qwen/Qwen2.5-0.5B", "Qwen2.5-0.5B"),
    ("Qwen/Qwen2.5-1.5B", "Qwen2.5-1.5B"),
    ("microsoft/Phi-3-mini-4k-instruct", "Phi-3-mini"),
]

# Four subjects rather than one, so the 100 questions spread across domains
# instead of being 100 questions of abstract algebra, where models this small
# sit at chance and the chart shows nothing.
MMLU_SUBJECTS = [
    "abstract_algebra",
    "high_school_world_history",
    "professional_medicine",
    "philosophy",
]

LETTERS = ["A", "B", "C", "D"]


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------

def mmlu_block(example, with_answer):
    """One MMLU question, optionally with its gold letter appended."""
    lines = [example["question"].strip()]
    for letter, choice in zip(LETTERS, example["choices"]):
        lines.append(f"{letter}. {choice}")
    gold = f" {LETTERS[example['answer']]}" if with_answer else ""
    lines.append(f"Answer:{gold}")
    return "\n".join(lines)


def build_mmlu(subjects, per_subject):
    """Return [(prompt, gold_letter_index)] with a per-subject 5-shot prefix."""
    items = []
    for subject in subjects:
        dev = load_dataset("cais/mmlu", subject, split="dev")  # exactly 5 examples
        test = load_dataset("cais/mmlu", subject, split=f"test[:{per_subject}]")
        pretty = subject.replace("_", " ")
        header = f"The following are multiple choice questions (with answers) about {pretty}.\n\n"
        shots = "\n\n".join(mmlu_block(ex, with_answer=True) for ex in dev)
        for ex in test:
            prompt = f"{header}{shots}\n\n{mmlu_block(ex, with_answer=False)}"
            items.append((prompt, ex["answer"]))
    return items


def gsm_gold(answer_text):
    """GSM8K gold answers end with a '####' marker followed by the number."""
    return answer_text.split("####")[-1].strip().replace(",", "")


CALC_ANNOTATION = re.compile(r"<" + r"<[^>]*>" + r">")


def build_gsm8k(n_examples, shots):
    train = load_dataset("openai/gsm8k", "main", split=f"train[:{shots}]")
    test = load_dataset("openai/gsm8k", "main", split=f"test[:{n_examples}]")
    blocks = []
    for ex in train:
        # Drop the bracketed calculator annotations; keep the reasoning.
        reasoning = CALC_ANNOTATION.sub("", ex["answer"].split("####")[0]).strip()
        blocks.append(
            f"Question: {ex['question'].strip()}\n"
            f"Answer: {reasoning}\nThe answer is {gsm_gold(ex['answer'])}."
        )
    prefix = "\n\n".join(blocks)
    items = []
    for ex in test:
        prompt = f"{prefix}\n\nQuestion: {ex['question'].strip()}\nAnswer:"
        items.append((prompt, gsm_gold(ex["answer"])))
    return items


NUMBER = re.compile(r"-?\d[\d,]*\.?\d*")
STATED = re.compile(r"[Tt]he answer is\s*\$?(-?\d[\d,]*\.?\d*)")
NEXT_QUESTION = re.compile(r"\n\s*Question:")


def canonical(value):
    """12, 12.0 and '12,000' should all compare equal."""
    try:
        number = float(str(value).replace(",", "").rstrip("."))
    except ValueError:
        return str(value)
    return str(int(number)) if number == int(number) else str(number)


def extract_number(text):
    """Prefer an explicit 'The answer is N'; otherwise take the last number."""
    tail = NEXT_QUESTION.split(text)[0]
    candidates = STATED.findall(tail) or NUMBER.findall(tail)
    return canonical(candidates[-1]) if candidates else None


# --------------------------------------------------------------------------
# evaluation
# --------------------------------------------------------------------------

@torch.no_grad()
def eval_mmlu(model, tokenizer, items, log_every):
    """Rank ' A'..' D' by log-probability - one forward pass per question."""
    choice_ids = [tokenizer.encode(f" {c}", add_special_tokens=False) for c in LETTERS]
    single_token = all(len(ids) == 1 for ids in choice_ids)
    if single_token:
        flat_ids = torch.tensor([ids[0] for ids in choice_ids], device=model.device)

    correct = 0
    prompt_tokens = 0
    start = time.perf_counter()
    for i, (prompt, gold) in enumerate(tqdm(items, desc="mmlu", leave=False)):
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        prompt_tokens += inputs["input_ids"].shape[1]
        if single_token:
            logits = model(**inputs).logits[0, -1].float()
            pred = int(torch.argmax(torch.log_softmax(logits, -1)[flat_ids]))
        else:
            scores = []
            for ids in choice_ids:
                suffix = torch.tensor([ids], device=model.device)
                full = torch.cat([inputs["input_ids"], suffix], dim=1)
                logprobs = torch.log_softmax(model(input_ids=full).logits[0].float(), -1)
                span = range(inputs["input_ids"].shape[1] - 1, full.shape[1] - 1)
                scores.append(sum(logprobs[p, t].item() for p, t in zip(span, ids)))
            pred = max(range(len(LETTERS)), key=scores.__getitem__)
        correct += int(pred == gold)
        if log_every and (i + 1) % log_every == 0:
            print(f"    mmlu {i + 1}/{len(items)} acc={correct / (i + 1):.3f}", flush=True)

    elapsed = time.perf_counter() - start
    return {
        "accuracy": correct / len(items),
        "tok_per_s": prompt_tokens / elapsed,
        "throughput_kind": "prefill",
        "examples": len(items),
        "seconds": elapsed,
    }


@torch.no_grad()
def eval_gsm8k(model, tokenizer, items, max_new_tokens, log_every, stop_on_question=False):
    correct = 0
    generated = 0
    records = []
    # Base models never emit EOS in a few-shot completion setting - they roll on
    # into a fabricated next question, so every sample runs to max_new_tokens.
    # Stopping at that boundary is the same answer for roughly half the compute,
    # but it changes how many tokens tok/s is averaged over, so it stays opt-in
    # and off by default for run-to-run comparability.
    stop_kwargs = (
        {"stop_strings": ["\nQuestion:"], "tokenizer": tokenizer} if stop_on_question else {}
    )
    start = time.perf_counter()
    for i, (prompt, gold) in enumerate(tqdm(items, desc="gsm8k", leave=False)):
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            **stop_kwargs,
        )
        new_tokens = out[0, inputs["input_ids"].shape[1]:]
        generated += new_tokens.shape[0]
        completion = tokenizer.decode(new_tokens, skip_special_tokens=True)
        pred = extract_number(completion)
        hit = pred is not None and pred == canonical(gold)
        correct += int(hit)
        records.append({"gold": gold, "pred": pred, "correct": hit})
        if log_every and (i + 1) % log_every == 0:
            print(f"    gsm8k {i + 1}/{len(items)} acc={correct / (i + 1):.3f}", flush=True)

    elapsed = time.perf_counter() - start
    metrics = {
        "accuracy": correct / len(items),
        "tok_per_s": generated / elapsed,
        "throughput_kind": "generation",
        "examples": len(items),
        "seconds": elapsed,
    }
    return metrics, records


def load_model(model_id, quant):
    # No trust_remote_code: transformers ships native Qwen2 and Phi3 classes, and
    # Phi-3's repo-bundled modeling_phi3.py targets transformers v4 (it raises
    # KeyError: 'type' under v5).
    kwargs = {"dtype": torch.bfloat16}
    if quant == "int4":
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        kwargs["device_map"] = {"": 0}
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    if quant != "int4":
        model = model.to("cuda" if torch.cuda.is_available() else "cpu")
    return model.eval()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quant", choices=["bf16", "int4"], default="int4")
    parser.add_argument("--mmlu-per-subject", type=int, default=25)
    parser.add_argument("--gsm8k-examples", type=int, default=100)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--models", nargs="*", default=None, help="filter by short name")
    parser.add_argument(
        "--stop-on-question",
        action="store_true",
        help="halt generation at the next fabricated 'Question:' (faster, opt-in)",
    )
    args = parser.parse_args()

    RESULTS.mkdir(exist_ok=True)
    csv_path = RESULTS / "benchmark.csv"
    if not torch.cuda.is_available():
        print("WARNING: no CUDA device visible - this will be very slow.", flush=True)

    print("Building prompts...", flush=True)
    mmlu_items = build_mmlu(MMLU_SUBJECTS, args.mmlu_per_subject)
    gsm_items = build_gsm8k(args.gsm8k_examples, shots=8)
    print(f"MMLU: {len(mmlu_items)} examples | GSM8K: {len(gsm_items)} examples", flush=True)

    models = [(mid, name) for mid, name in MODELS if not args.models or name in args.models]

    # Re-running one model must not discard the others' finished rows: keep any
    # prior results except for the models this invocation is about to replace.
    prior = pd.DataFrame()
    if csv_path.exists():
        prior = pd.read_csv(csv_path)
        prior = prior[~prior["model"].isin([name for _, name in models])]
        if len(prior):
            print(f"Keeping {len(prior)} prior row(s) for "
                  f"{', '.join(sorted(set(prior['model'])))}", flush=True)

    rows = []

    def save():
        pd.concat([prior, pd.DataFrame(rows)], ignore_index=True).to_csv(csv_path, index=False)
    for model_id, name in models:
        print(f"\n=== {name} ({args.quant}) ===", flush=True)
        load_start = time.perf_counter()
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = load_model(model_id, args.quant)
        print(f"loaded in {time.perf_counter() - load_start:.1f}s", flush=True)

        for task in ("mmlu", "gsm8k"):
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            if task == "mmlu":
                metrics = eval_mmlu(model, tokenizer, mmlu_items, args.log_every)
            else:
                metrics, records = eval_gsm8k(
                    model, tokenizer, gsm_items, args.max_new_tokens, args.log_every,
                    stop_on_question=args.stop_on_question,
                )
                (RESULTS / f"gsm8k_{name}.json").write_text(json.dumps(records, indent=2))
            metrics["peak_vram_gb"] = (
                torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0.0
            )
            rows.append({"model": name, "task": task, "quant": args.quant, **metrics})
            print(
                f"  {task}: acc={metrics['accuracy']:.3f} "
                f"{metrics['tok_per_s']:.1f} tok/s ({metrics['throughput_kind']}) "
                f"peak={metrics['peak_vram_gb']:.2f} GB in {metrics['seconds']:.0f}s",
                flush=True,
            )
            # Save after every task so a crash never loses finished work.
            save()

        del model
        gc.collect()
        torch.cuda.empty_cache()

    save()
    print("\n=== Results ===")
    print(pd.read_csv(csv_path).to_string(index=False))
    print(f"\nWrote {csv_path}")


if __name__ == "__main__":
    main()
