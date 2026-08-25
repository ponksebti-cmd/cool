import torch
from model import Transformer, ModelArgs

def test_generation():
    # Use the TINY_TEST equivalent config
    args = ModelArgs(
        dim=256,
        n_layers=4,
        n_heads=8,
        n_kv_heads=2,
        vocab_size=32000,
        max_seq_len=256
    )
    
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
        
    print(f"Loading untrained model on {device}...")
    model = Transformer(args).to(device)
    model.eval()
    
    # We don't have a tokenizer, so we simulate the word "hi" with random token IDs
    # Let's say "hi" is represented by token IDs [104, 105]
    print("Simulating prompt: 'hi' -> Token IDs: [104, 105]")
    input_ids = torch.tensor([[104, 105]], dtype=torch.long, device=device)
    
    # Generate 10 new tokens manually
    print("Generating 10 new tokens...")
    output_ids = input_ids
    for _ in range(10):
        logits = model(output_ids)
        next_token_logits = logits[:, -1, :] / 0.8  # Apply temperature
        
        # Top-K
        v, _ = torch.topk(next_token_logits, 50)
        next_token_logits[next_token_logits < v[:, [-1]]] = -float('Inf')
        
        probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        output_ids = torch.cat((output_ids, next_token), dim=1)
        
    print("\n--- Output ---")
    print(f"Input Tokens: {input_ids[0].tolist()}")
    print(f"Generated Sequence: {output_ids[0].tolist()}")
    print("-------------")
    print("Note: Since the model is randomly initialized and untrained, the generated tokens are completely random gibberish.")

if __name__ == "__main__":
    test_generation()
