"""Compare fixed-token Gemma 4 runs on Splash and the installed llama.cpp."""

import argparse
import json
import math
import subprocess
from array import array
from heapq import nlargest
from pathlib import Path

from install.autoregressive import chat_prompt


PROMPTS = {
    "japan": "What is the capital of Japan? Answer with just the city name.",
    "arithmetic": "What is 17 + 25? Answer with only the number.",
    "long": (
        "The code word at the very beginning of these notes is cobalt.\n"
        + "The field notes record ordinary measurements from the same lab.\n" * 90
        + "What was the code word at the very beginning? Reply with only the word."
    ),
}


def invoke(command, log, data=None):
    with log.open("wb") as errors:
        return subprocess.run(
            command, input=data, stdout=subprocess.PIPE, stderr=errors, check=True, timeout=900
        ).stdout


def logits(path):
    values = array("f")
    values.frombytes(path.read_bytes())
    if len(values) != 262144 or not all(map(math.isfinite, values)):
        raise ValueError(f"invalid logits in {path}")
    return values


def compare(splash, reference, splash_tokens, reference_tokens):
    differences = [a - b for a, b in zip(splash, reference, strict=True)]
    top_splash = nlargest(10, range(len(splash)), key=splash.__getitem__)
    top_reference = nlargest(10, range(len(reference)), key=reference.__getitem__)
    matched = 0
    for native_id, reference_id in zip(splash_tokens, reference_tokens):
        if native_id != reference_id:
            break
        matched += 1
    return {
        "max_abs_logit_error": max(map(abs, differences)),
        "rms_logit_error": math.sqrt(math.fsum(x * x for x in differences) / len(differences)),
        "top10_overlap": len(set(top_splash) & set(top_reference)),
        "splash_top10": [(token, splash[token]) for token in top_splash],
        "reference_top10": [(token, reference[token]) for token in top_reference],
        "greedy_common_prefix": matched,
        "greedy_identical": splash_tokens == reference_tokens,
    }


def run_case(model, directory, name):
    tool = Path("build/gguf-text")
    tokens = directory / f"{name}.ids"
    tokens.write_bytes(
        invoke([tool, "tokenize", model], directory / f"{name}.tokenizer.log",
               chat_prompt(PROMPTS[name]).encode())
    )
    count = len(tokens.read_bytes().split())
    if name == "long" and count <= 1024:
        raise ValueError("the long prompt did not cross the sliding window")
    native_logits = directory / f"{name}.splash.logits"
    ref_logits = directory / f"{name}.reference.logits"
    native = invoke(
        ["build/splash", "autoregressive", model, "build/splash.metallib", tokens, "8", native_logits],
        directory / f"{name}.splash.log",
    )
    reference = invoke(
        [tool, "reference", model, tokens, "8", ref_logits], directory / f"{name}.reference.log"
    )
    (directory / f"{name}.splash.json").write_bytes(native)
    (directory / f"{name}.reference.json").write_bytes(reference)
    native, reference = json.loads(native), json.loads(reference)
    for label, result in (("splash", native), ("reference", reference)):
        ids = "\n".join(map(str, result["tokens"])).encode()
        result["text"] = invoke(
            [tool, "detokenize", model], directory / f"{name}.{label}.decode.log", ids
        ).decode(errors="replace")
    native["prefill_tokens_per_second"] = count / native["prefill_seconds"]
    if native["decode_rows"]:
        native["decode_tokens_per_second"] = native["decode_rows"] / native["decode_seconds"]
    scores = compare(logits(native_logits), logits(ref_logits), native["tokens"], reference["tokens"])
    scores["passed"] = (
        scores["top10_overlap"] >= 8
        and scores["rms_logit_error"] <= 0.15
        and scores["splash_top10"][0][0] == scores["reference_top10"][0][0]
        and scores["greedy_identical"]
    )
    return {"prompt_tokens": count, "splash": native, "reference": reference, **scores}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--output", type=Path, default=Path("build/gemma4-validation"))
    parser.add_argument("--case", action="append", choices=PROMPTS)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    results = {}
    for name in args.case or PROMPTS:
        results[name] = run_case(args.model.resolve(), args.output, name)
        (args.output / "parity.json").write_text(json.dumps(results, indent=2) + "\n")
        print(json.dumps({name: results[name]}), flush=True)
    failed = [name for name, result in results.items() if not result["passed"]]
    if failed:
        raise SystemExit("parity failed: " + ", ".join(failed))


if __name__ == "__main__":
    main()
