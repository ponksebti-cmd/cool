import torch
import torch.nn.functional as F
from model import Transformer, ModelArgs
from data import get_dummy_dataloader

def overfit_single_batch():
    # Use 8 layers to ensure we hit both Mamba (0,1,2, 4,5,6) and Attn (3, 7) layers
    args = ModelArgs(
        dim=256,
        n_layers=8,
        n_heads=8,
        n_kv_heads=2,
        vocab_size=32000,
        max_seq_len=256,
        hybrid_pattern="3:1"
    )
    
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
        
    print(f"Instantiating Hybrid model (Mamba-2 + Attention) on {device}...")
    model = Transformer(args).to(device)
    
    # Verify architecture
    mamba_count = sum(1 for l in model.layers if not hasattr(l.attention, 'wq'))
    attn_count = sum(1 for l in model.layers if hasattr(l.attention, 'wq'))
    print(f"Architecture: {mamba_count} Mamba-2 blocks, {attn_count} Attention blocks.")
    
    # Overfit to a single batch
    dataloader = get_dummy_dataloader(batch_size=2, max_seq_len=256, vocab_size=args.vocab_size)
    batch = next(iter(dataloader))
    
    tokens, mask, labels = batch
    tokens = tokens.to(device)
    mask = mask.to(device)
    labels = labels.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    print("Starting hybrid overfit test...")
    model.train()
    for step in range(250):
        optimizer.zero_grad()
        
        logits = model(tokens, mask)
        
        loss = F.cross_entropy(logits.view(-1, args.vocab_size), labels.view(-1), ignore_index=-100)
        
        loss.backward()
        optimizer.step()
        
        if step % 25 == 0 or step == 249:
            print(f"Step {step:03d} | Loss: {loss.item():.4f}")
            
    if loss.item() < 0.1:
        print("\nSUCCESS: Hybrid model successfully overfit the single batch to near-zero loss!")
    else:
        print("\nFAILURE: Model failed to overfit. Loss is still too high.")

if __name__ == "__main__":
    overfit_single_batch()
