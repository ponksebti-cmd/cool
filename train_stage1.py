import torch
import torch.nn.functional as F
from model import Transformer, ModelArgs
from data import get_dummy_dataloader

def overfit_single_batch():
    # Use a small configuration for fast local validation
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
    
    print(f"Instantiating model on {device}...")
    model = Transformer(args).to(device)
    
    # Overfit to a single batch
    dataloader = get_dummy_dataloader(batch_size=2, max_seq_len=256, vocab_size=args.vocab_size)
    batch = next(iter(dataloader))
    
    tokens, mask, labels = batch
    tokens = tokens.to(device)
    mask = mask.to(device)
    labels = labels.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    print("Starting overfit test...")
    model.train()
    for step in range(150):
        optimizer.zero_grad()
        
        logits = model(tokens, mask)
        
        # Flatten logits and labels for CrossEntropyLoss
        loss = F.cross_entropy(logits.view(-1, args.vocab_size), labels.view(-1), ignore_index=-100)
        
        loss.backward()
        optimizer.step()
        
        if step % 20 == 0 or step == 149:
            print(f"Step {step:03d} | Loss: {loss.item():.4f}")
            
    if loss.item() < 0.1:
        print("\nSUCCESS: Model successfully overfit the single batch to near-zero loss!")
    else:
        print("\nFAILURE: Model failed to overfit. Loss is still too high.")

if __name__ == "__main__":
    overfit_single_batch()
