import torch

class PackedDataloaderWrapper:
    def __init__(self, dataloader, max_seq_len):
        self.dataloader = dataloader
        self.max_seq_len = max_seq_len

    def __iter__(self):
        for batch in self.dataloader:
            # batch is expected to be a list of token lists (documents)
            # For simplicity, this wrapper packs documents into sequences of max_seq_len
            # and generates the block-diagonal attention mask
            yield self.pack_documents(batch)
            
    def pack_documents(self, documents):
        """
        Packs multiple documents into a single sequence of max_seq_len.
        Creates a block-diagonal attention mask to prevent cross-document attention.
        """
        bsz = len(documents)
        packed_tokens = torch.zeros((bsz, self.max_seq_len), dtype=torch.long)
        masks = torch.zeros((bsz, 1, self.max_seq_len, self.max_seq_len), dtype=torch.bool)
        labels = torch.zeros((bsz, self.max_seq_len), dtype=torch.long) - 100 # -100 for ignore index
        
        for b, docs in enumerate(documents):
            # docs is a list of token arrays for a single batch element
            # We pack them sequentially
            curr_len = 0
            for doc in docs:
                doc_len = len(doc)
                if curr_len + doc_len > self.max_seq_len:
                    doc = doc[:self.max_seq_len - curr_len]
                    doc_len = len(doc)
                
                if doc_len <= 1:
                    continue
                    
                # Fill tokens
                packed_tokens[b, curr_len:curr_len+doc_len] = torch.tensor(doc)
                
                # Fill labels (shifted by 1)
                labels[b, curr_len:curr_len+doc_len-1] = torch.tensor(doc[1:])
                
                # Create causal mask for this document block
                block_mask = torch.tril(torch.ones((doc_len, doc_len), dtype=torch.bool))
                masks[b, 0, curr_len:curr_len+doc_len, curr_len:curr_len+doc_len] = block_mask
                
                curr_len += doc_len
                if curr_len >= self.max_seq_len:
                    break
                    
        return packed_tokens, masks, labels

def get_dummy_dataloader(batch_size=2, max_seq_len=256, vocab_size=32000):
    # Generates some random documents
    dataset = []
    for _ in range(10): # 10 batches
        batch = []
        for _ in range(batch_size):
            # 2 documents per batch element
            docs = [
                torch.randint(0, vocab_size, (max_seq_len // 2,)).tolist(),
                torch.randint(0, vocab_size, (max_seq_len // 2,)).tolist()
            ]
            batch.append(docs)
        dataset.append(batch)
        
    return PackedDataloaderWrapper(dataset, max_seq_len)
