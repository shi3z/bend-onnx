#!/usr/bin/env python3
"""
nanoGPT Text Generation CLI in Bend / PyTorch Reference.
Allows custom prompts, token length, and runs either reference PyTorch inference or pure Bend inference.
"""

import os
import sys
import argparse
import subprocess
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from nanogpt.train_and_export import CHARS, VOCAB_SIZE, encode, decode, export_to_bend, train_model

BEND_BIN = os.path.expanduser("~/.bend/bin/bend")

def main():
    parser = argparse.ArgumentParser(description="nanoGPT text generation runner for Bend")
    parser.add_argument("--prompt", type=str, default="BEND ", help="Initial text prompt (chars: A-Z, space, !, ., :, 0, 1)")
    parser.add_argument("--steps", type=int, default=10, help="Number of tokens to generate")
    parser.add_argument("--backend", choices=["bend", "torch"], default="bend", help="Execution backend")
    args = parser.parse_args()

    # Clean and uppercase prompt
    prompt = args.prompt.upper()
    valid_prompt = "".join([c for c in prompt if c in CHARS])
    if not valid_prompt:
        valid_prompt = "BEND "

    print(f"=== nanoGPT Inference ===")
    print(f"Prompt:  '{valid_prompt}'")
    print(f"Steps:   {args.steps}")
    print(f"Backend: {args.backend}")
    print("=========================")

    if args.backend == "torch":
        model, _ = train_model()
        model.eval()
        cur_ids = encode(valid_prompt)
        for _ in range(args.steps):
            x = torch.tensor([cur_ids], dtype=torch.long)
            logits = model(x)
            next_id = int(torch.argmax(logits[0, -1]).item())
            cur_ids.append(next_id)
        print(f"Generated Result: '{decode(cur_ids)}'")
    else:
        bend_path = os.path.join(PROJECT_ROOT, "nanogpt", "nanogpt.bend")
        if not os.path.exists(bend_path):
            print("nanogpt.bend not found. Training and generating...")
            model, config = train_model()
            export_to_bend(model, config, bend_path)
        
        print("Running pure Bend runtime:")
        res = subprocess.run([BEND_BIN, bend_path], capture_output=True, text=True)
        if res.returncode != 0:
            print(f"Bend execution error:\n{res.stderr}")
            sys.exit(1)
        print(res.stdout)

if __name__ == '__main__':
    main()
