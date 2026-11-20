"""Generate one text response from a local GGUF through the native engine."""

import json
import subprocess
import tempfile
from pathlib import Path


def chat_prompt(prompt):
    prompt = prompt.strip()
    markers = ("<|turn>", "<turn|>", "<|channel>", "<channel|>")
    if not prompt or any(mark in prompt for mark in markers):
        raise OSError("expected a text prompt without chat control markers")
    return f"<|turn>user\n{prompt}<turn|>\n<|turn>model\n<|channel>thought\n<channel|>"


def generate(args):
    build = Path(__file__).resolve().parents[1] / "build"
    native, library, tokenizer = (build / name for name in ("splash", "splash.metallib", "gguf-text"))
    if not native.is_file() or not library.is_file():
        raise OSError("build the native Splash runtime first")
    if not tokenizer.is_file():
        raise OSError("build the GGUF tokenizer with make build/gguf-text first")
    model = args.model.expanduser().resolve()
    if not model.is_file():
        raise OSError("--model must name a local GGUF file")
    if not 1 <= args.max_tokens < 4096:
        raise OSError("--max-tokens must be between 1 and 4095")
    if not 0 <= args.lookup_tokens <= 31:
        raise OSError("--lookup-tokens must be between 0 and 31")
    draft = []
    if args.draft is not None:
        checkpoint = args.draft.expanduser().resolve()
        if not (checkpoint / "config.json").is_file():
            raise OSError("--draft must name a local DFlash draft directory with its config.json")
        draft = ["--draft", checkpoint]
    prompt = chat_prompt(args.prompt)
    try:
        subprocess.run([native, "autoregressive-check", model], check=True)
        ids = subprocess.check_output([tokenizer, "tokenize", model], input=prompt.encode())
        with tempfile.TemporaryDirectory(prefix="text-", dir=build) as directory:
            tokens = Path(directory) / "prompt.ids"
            tokens.write_bytes(ids)
            response = json.loads(
                subprocess.check_output(
                    [native, "autoregressive", model, library, tokens, str(args.max_tokens),
                     "--lookup-tokens", str(args.lookup_tokens), *draft]
                )
            )
        output_ids = "\n".join(map(str, response["tokens"]))
        output = subprocess.check_output([tokenizer, "detokenize", model], input=output_ids.encode())
    except subprocess.CalledProcessError as error:
        raise OSError(f"text generation command failed with exit status {error.returncode}") from error
    print(output.decode(errors="replace"))
    return 0
