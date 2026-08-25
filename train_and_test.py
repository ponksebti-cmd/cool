#!/usr/bin/env python3
"""
train_and_test.py — Train on real English text, then generate readable output.

Strategy: byte-level (character) encoding.
  - Every character becomes its UTF-8 byte value (0-255).
  - TINY_TEST config has vocab_size=256, which is a perfect match.
  - No external tokenizer library needed.
  - After ~500 steps on ~4 KB of Shakespeare, the model learns character
    n-gram patterns and produces visibly structured (not random) English.

Usage:
    python3 train_and_test.py
    python3 train_and_test.py --steps 800 --prompt "Friends"
"""

from __future__ import annotations
import sys
import math
import time
import argparse
from contextlib import nullcontext

import torch
import torch.nn.functional as F

sys.path.insert(0, ".")
from src import Transformer, TINY_TEST, make_dataloader

# ── Embedded text corpus (~4 KB of public-domain Shakespeare) ─────────────────
CORPUS = """\
To be, or not to be, that is the question:
Whether 'tis nobler in the mind to suffer
The slings and arrows of outrageous fortune,
Or to take arms against a sea of troubles
And by opposing end them. To die, to sleep,
No more; and by a sleep to say we end
The heart-ache and the thousand natural shocks
That flesh is heir to: 'tis a consummation
Devoutly to be wish'd. To die, to sleep;
To sleep, perchance to dream, ay, there's the rub:
For in that sleep of death what dreams may come,
When we have shuffled off this mortal coil,
Must give us pause.

All the world's a stage,
And all the men and women merely players;
They have their exits and their entrances,
And one man in his time plays many parts.
At first, the infant, mewling and puking in the nurse's arms.
Then the whining schoolboy, with his satchel
And shining morning face, creeping like snail
Unwillingly to school. And then the lover,
Sighing like furnace, with a woeful ballad
Made to his mistress' eyebrow. Then a soldier,
Full of strange oaths and bearded like the pard,
Jealous in honour, sudden and quick in quarrel,
Seeking the bubble reputation even in the cannon's mouth.

Friends, Romans, countrymen, lend me your ears;
I come to bury Caesar, not to praise him.
The evil that men do lives after them;
The good is oft interred with their bones;
So let it be with Caesar. The noble Brutus
Hath told you Caesar was ambitious:
If it were so, it was a grievous fault,
And grievously hath Caesar answer'd it.
Here, under leave of Brutus and the rest,
For Brutus is an honourable man;
So are they all, all honourable men,
Come I to speak in Caesar's funeral.
He was my friend, faithful and just to me:
But Brutus says he was ambitious;
And Brutus is an honourable man.

Two households, both alike in dignity,
In fair Verona, where we lay our scene,
From ancient grudge break to new mutiny,
Where civil blood makes civil hands unclean.
From forth the fatal loins of these two foes
A pair of star-cross'd lovers take their life;
The fearful passage of their death-mark'd love,
And the continuance of their parents' rage,
Which but their children's end nought could remove,
Is now the two hours traffic of our stage.

What a piece of work is a man, how noble in reason,
how infinite in faculty, in form and moving how express
and admirable, in action how like an angel, in apprehension
how like a god, the beauty of the world, the paragon of animals.
And yet to me what is this quintessence of dust?

O Romeo, Romeo, wherefore art thou Romeo?
Deny thy father and refuse thy name;
Or if thou wilt not, be but sworn my love,
And I'll no longer be a Capulet.
'Tis but thy name that is my enemy;
Thou art thyself, though not a Montague.
What's Montague? It is nor hand, nor foot,
Nor arm, nor face, nor any other part
Belonging to a man. O be some other name!
What's in a name? That which we call a rose
By any other name would smell as sweet;
So Romeo would, were he not Romeo call'd,
Retain that dear perfection which he owes
Without that title.

Now is the winter of our discontent
Made glorious summer by this sun of York;
And all the clouds that lour'd upon our house
In the deep bosom of the ocean buried.
Now are our brows bound with victorious wreaths;
Our bruised arms hung up for monuments;
Our stern alarums changed to merry meetings,
Our dreadful marches to delightful measures.
Grim-visaged war hath smooth'd his wrinkled front;
And now, instead of mounting barbed steeds
To fright the souls of fearful adversaries,
He capers nimbly in a lady's chamber
To the lascivious pleasing of a lute.
"""


# ── Tokeniser (byte-level) ────────────────────────────────────────────────────

def encode(text: str) -> list[int]:
    """String → list of byte values (0-255)."""
    return list(text.encode("utf-8"))


def decode(tokens: list[int]) -> str:
    """List of byte values → string (replace invalid bytes)."""
    return bytes([max(0, min(255, t)) for t in tokens]).decode("utf-8", errors="replace")


# ── Data preparation ──────────────────────────────────────────────────────────

def make_documents(
    text: str,
    doc_len: int = 64,
    repeat: int = 8,
) -> list[list[int]]:
    """
    Split text into short documents and repeat the corpus several times.
    Repetition ensures enough gradient steps even with a tiny dataset.
    """
    tokens = encode(text)
    docs = []
    for _ in range(repeat):
        for i in range(0, len(tokens) - doc_len, doc_len // 2):  # 50% overlap
            chunk = tokens[i : i + doc_len]
            if len(chunk) >= 8:
                docs.append(chunk)
    return docs


# ── Training ──────────────────────────────────────────────────────────────────

def train(
    steps: int = 600,
    lr: float = 3e-3,
    batch_size: int = 8,
    log_every: int = 50,
) -> Transformer:
    """Train the TINY model on the Shakespeare corpus. Returns trained model."""

    # Device
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    dev = torch.device(device)

    # AMP context (bfloat16 on CUDA/MPS)
    use_amp = device in ("cuda", "mps")
    amp_ctx = (
        torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=use_amp)
        if use_amp else nullcontext()
    )

    # Config — TINY_TEST has vocab_size=256 which matches byte encoding perfectly
    config = TINY_TEST

    print(f"\n{'═'*62}")
    print(f"  Training on Shakespeare corpus — byte-level encoding")
    print(f"{'═'*62}")
    print(f"  Device    : {dev}")
    print(f"  AMP       : {'bfloat16 ✓' if use_amp else 'float32'}")
    print(f"  Vocab     : 256 (all UTF-8 bytes)")
    print(f"  Config    : TINY  ({config.n_layers}L × {config.hidden_dim}d)")
    print(f"  Steps     : {steps}")
    print(f"  Baseline  : {math.log(256):.2f}  (random-init cross-entropy)")
    print(f"{'═'*62}\n")

    # Model
    torch.manual_seed(42)
    model = Transformer(config).to(dev)
    n_params = model.num_parameters()
    print(f"  Parameters: {n_params:,} ({n_params/1e6:.2f}M)\n")

    # Data
    docs   = make_documents(CORPUS, doc_len=64, repeat=12)
    loader = make_dataloader(
        documents=docs,
        block_size=config.block_size,
        batch_size=batch_size,
        shuffle=True,
        seed=42,
    )
    data_iter = iter(loader)

    def next_batch():
        nonlocal data_iter
        try:
            return next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            return next(data_iter)

    # Optimizer — AdamW with mild weight decay
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.05, eps=1e-8
    )

    # Cosine decay after warmup
    warmup   = steps // 10
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=steps, eta_min=lr * 0.1
    )

    # Training loop
    model.train()
    t0 = time.perf_counter()
    best_loss = float("inf")

    for step in range(1, steps + 1):
        # Linear warmup
        if step <= warmup:
            for pg in optimizer.param_groups:
                pg["lr"] = lr * (step / warmup)

        batch        = next_batch()
        input_ids    = batch["input_ids"].to(dev, non_blocking=True)
        targets      = batch["targets"].to(dev, non_blocking=True)
        position_ids = batch["position_ids"].to(dev, non_blocking=True)
        attn_mask    = batch["attn_mask"].to(dev, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with amp_ctx:
            _, loss = model(
                input_ids,
                attn_mask=attn_mask,
                position_ids=position_ids,
                targets=targets,
            )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step > warmup:
            scheduler.step()

        if loss.item() < best_loss:
            best_loss = loss.item()

        if step % log_every == 0 or step == 1:
            elapsed = time.perf_counter() - t0
            current_lr = optimizer.param_groups[0]["lr"]
            # Show bits-per-character (bpc) = loss / log(2), easier to interpret
            bpc = loss.item() / math.log(2)
            bar_width = 20
            filled = int(bar_width * step / steps)
            bar = "█" * filled + "░" * (bar_width - filled)
            print(
                f"  [{bar}] step {step:4d}/{steps}"
                f"  loss: {loss.item():.4f}  bpc: {bpc:.2f}"
                f"  lr: {current_lr:.1e}  ({elapsed:.1f}s)"
            )

    total = time.perf_counter() - t0
    print(f"\n  ✓ Done in {total:.1f}s  |  Best loss: {best_loss:.4f}"
          f"  |  bits/char: {best_loss/math.log(2):.2f}\n")

    return model


# ── Generation ────────────────────────────────────────────────────────────────

def generate_text(
    model: Transformer,
    prompt: str,
    max_new_tokens: int = 300,
    temperature: float = 0.8,
    top_k: int = 40,
) -> str:
    """Encode prompt, run autoregressive generation, decode back to text."""

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    dev = torch.device(device)

    model.eval()
    prompt_ids  = encode(prompt)
    input_ids   = torch.tensor([prompt_ids], dtype=torch.long, device=dev)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
        )

    # Decode only the newly generated tokens (after the prompt)
    new_tokens = output_ids[0, len(prompt_ids):].tolist()
    return decode(new_tokens)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--steps",       type=int,   default=600)
    p.add_argument("--lr",          type=float, default=3e-3)
    p.add_argument("--batch-size",  type=int,   default=8)
    p.add_argument("--prompt",      type=str,   default=None,
                   help="Custom generation seed text")
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--tokens",      type=int,   default=300,
                   help="Number of characters to generate")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ── Train ─────────────────────────────────────────────────────────────
    model = train(
        steps=args.steps,
        lr=args.lr,
        batch_size=args.batch_size,
    )

    # ── Generate from multiple prompts ────────────────────────────────────
    prompts = [
        args.prompt if args.prompt else "To be, or not to be",
        "Friends, Romans",
        "What's in a name?",
        "Now is the winter",
    ]
    if args.prompt:
        prompts = [args.prompt]  # only user prompt if specified

    print(f"{'═'*62}")
    print(f"  Generation (temp={args.temperature}, top_k=40)")
    print(f"{'═'*62}")

    for prompt in prompts:
        generated = generate_text(
            model,
            prompt=prompt,
            max_new_tokens=args.tokens,
            temperature=args.temperature,
        )
        print(f"\n  ┌─ Prompt: {repr(prompt)}")
        print(f"  │")
        # Pretty-print: indent each line of output
        full_text = prompt + generated
        for line in full_text.splitlines():
            print(f"  │  {line}")
        print(f"  └{'─'*58}")

    print()


if __name__ == "__main__":
    main()
