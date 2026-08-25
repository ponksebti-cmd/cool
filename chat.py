#!/usr/bin/env python3
"""
chat.py — Interactive terminal chat for your SFT model.

Usage:
    python3 chat.py --config 300m --ckpt sft_ckpt_final.pt
"""

from __future__ import annotations
import argparse
import sys
import torch
from transformers import AutoTokenizer

sys.path.insert(0, ".")
from src import Transformer, DEFAULT_300M, TINY_TEST


def generate_response(
    model: Transformer,
    tokenizer: AutoTokenizer,
    prompt: str,
    device: torch.device,
    max_new_tokens: int = 200,
    temperature: float = 0.7,
    top_k: int = 40,
) -> str:
    model.eval()
    
    # We must format the prompt EXACTLY how we trained it in train_sft.py
    formatted_prompt = f"USER: {prompt}\nASSISTANT: "
    
    prompt_ids = tokenizer.encode(formatted_prompt, add_special_tokens=True)
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
        )
        
    # Slice off the prompt to only return the new generated tokens
    new_tokens = output_ids[0, len(prompt_ids):].tolist()
    
    # Stop generation at EOS token if the model emitted one
    if tokenizer.eos_token_id in new_tokens:
        eos_idx = new_tokens.index(tokenizer.eos_token_id)
        new_tokens = new_tokens[:eos_idx]
        
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="300m", choices=["tiny", "300m"])
    p.add_argument("--ckpt", type=str, required=True, help="Path to SFT checkpoint")
    p.add_argument("--device", default="auto")
    args = p.parse_args()
    
    device = args.device
    if device == "auto":
        if torch.cuda.is_available(): device = "cuda"
        elif torch.backends.mps.is_available(): device = "mps"
        else: device = "cpu"
    dev = torch.device(device)

    print("\n[Chat] Loading Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")

    config = TINY_TEST if args.config == "tiny" else DEFAULT_300M
    print(f"[Chat] Loading Model architecture on {dev}...")
    model = Transformer(config).to(dev)
    
    print(f"[Chat] Loading Weights from {args.ckpt}...")
    ckpt = torch.load(args.ckpt, map_location=dev)
    model.load_state_dict(ckpt["model_state_dict"])
    
    print("\n" + "="*50)
    print("  AI Assistant is Ready. Type 'quit' to exit.")
    print("="*50 + "\n")
    
    while True:
        try:
            user_input = input("You: ")
            if user_input.lower() in ["quit", "exit"]:
                break
            if not user_input.strip():
                continue
                
            print("Assistant: ", end="", flush=True)
            # In a fully optimized script, we would yield tokens one-by-one.
            # Here we generate the whole response then print it.
            response = generate_response(model, tokenizer, user_input, dev)
            print(response + "\n")
            
        except KeyboardInterrupt:
            break
            
    print("\nGoodbye!")

if __name__ == "__main__":
    main()
