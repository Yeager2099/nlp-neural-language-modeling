# Install required packages (uncomment if needed)
# !pip install torch numpy matplotlib requests

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import requests
import time

# Set random seeds for reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# Device detection
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")
if device == 'cpu':
    print("NOTE: Training the transformer on CPU takes ~15-30 minutes.")
    print("      If you can use a GPU, training drops to ~2-3 minutes.")
    print("      In Colab: Runtime -> Change runtime type -> T4 GPU (free).")

print(f"PyTorch version: {torch.__version__}")
print("All imports successful.")

########## DO NOT MODIFY ##########

# Load Tiny Shakespeare dataset
url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
corpus = requests.get(url).text

print(f"Corpus length: {len(corpus):,} characters")
print(f"\nFirst 200 characters:")
print(corpus[:200])

########## DO NOT MODIFY ##########

# Model hyperparameters
batch_size = 32        # Number of sequences in each training batch
block_size = 64        # Context window length (number of tokens the model sees at once)
n_embd = 128           # Embedding dimension
n_head = 4             # Number of attention heads in multi-head attention
n_layer = 4            # Number of stacked transformer blocks
dropout_rate = 0.1     # Dropout probability for regularization
learning_rate = 3e-4   # Learning rate for AdamW optimizer

print("Hyperparameters:")
print(f"  batch_size  = {batch_size}")
print(f"  block_size  = {block_size}")
print(f"  n_embd      = {n_embd}")
print(f"  n_head      = {n_head}")
print(f"  n_layer     = {n_layer}")
print(f"  dropout     = {dropout_rate}")
print(f"  lr          = {learning_rate}")
print(f"  head_size   = n_embd // n_head = {n_embd // n_head}")

## YOUR CODE HERE

# Create a sorted list of all unique characters in the corpus
vocab = sorted(list(set(corpus)))
vocab_size = len(vocab)

# Create character-to-index mapping
stoi = {ch: i for i, ch in enumerate(vocab)}

# Create index-to-character mapping
itos = {i: ch for i, ch in enumerate(vocab)}

# Encode: string -> list of integers
def encode(s):
    """Convert a string to a list of integer token IDs.
    
    Args:
        s: A string composed of characters in the vocabulary.
    
    Returns:
        A list of integers, one per character.
    """
    return [stoi[c] for c in s]

# Decode: list of integers -> string
def decode(int_list):
    """Convert a list of integer token IDs back to a string.
    
    Args:
        int_list: A list of integers representing token IDs.
    
    Returns:
        A string.
    """
    return ''.join([itos[i] for i in int_list])

# Sanity checks for tokenizer

assert vocab is not None and len(vocab) > 0, "vocab should be a non-empty list"
assert vocab == sorted(vocab), "vocab should be sorted"
assert vocab_size == len(vocab), f"vocab_size should equal len(vocab), got {vocab_size} vs {len(vocab)}"
assert len(stoi) == vocab_size, "stoi should have one entry per character"
assert len(itos) == vocab_size, "itos should have one entry per character"

# Round-trip test
test_str = "hi there"
assert decode(encode(test_str)) == test_str, \
    f"encode/decode round-trip failed: got '{decode(encode(test_str))}' instead of '{test_str}'"

# Check specific encoding (Tiny Shakespeare has a known character set)
assert encode("hi there") == [46, 47, 1, 58, 46, 43, 56, 43], \
    f"Unexpected encoding for 'hi there': {encode('hi there')}"

# Full corpus round-trip
assert decode(encode(corpus[:100])) == corpus[:100], "Full corpus round-trip failed"

print(f"Vocabulary size: {vocab_size} characters")
print(f"Characters: {''.join(vocab)}")
print(f"encode('hi there') = {encode('hi there')}")
print(f"decode([46, 47, 1, 58, 46, 43, 56, 43]) = '{decode([46, 47, 1, 58, 46, 43, 56, 43])}'")
print("\nTokenizer sanity checks passed!")

########## DO NOT MODIFY ##########

# Encode the entire corpus as a tensor
data = torch.tensor(encode(corpus), dtype=torch.long)
print(f"Encoded corpus: {data.shape[0]:,} tokens")

# Train/val split: first 90% for training, last 10% for validation
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

print(f"Training tokens:   {len(train_data):,}")
print(f"Validation tokens: {len(val_data):,}")

def get_batch(split):
    """Sample a random mini-batch of (input, target) pairs.
    
    Args:
        split: 'train' or 'val' -- which dataset to sample from.
    
    Returns:
        x: Tensor of shape (batch_size, block_size) -- input token sequences.
        y: Tensor of shape (batch_size, block_size) -- target token sequences (shifted by 1).
    
    Steps:
        1. Select train_data or val_data based on the split argument.
        2. Sample batch_size random starting indices.
           Each index i should satisfy: 0 <= i <= len(data) - block_size - 1
           (we need block_size + 1 tokens starting from i to create both x and y).
        3. For each starting index i:
           - x[j] = data[i : i + block_size]
           - y[j] = data[i+1 : i + block_size + 1]
        4. Stack into tensors and move to device.
    
    Hint: Use torch.randint for sampling and torch.stack for batching.
    """
    ## YOUR CODE HERE
    data_split = train_data if split == 'train' else val_data
    ix = torch.randint(0, len(data_split) - block_size, (batch_size,))
    x = torch.stack([data_split[i:i+block_size] for i in ix])
    y = torch.stack([data_split[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

# Sanity checks for get_batch

xb, yb = get_batch('train')

assert xb.shape == (batch_size, block_size), \
    f"x shape should be ({batch_size}, {block_size}), got {xb.shape}"
assert yb.shape == (batch_size, block_size), \
    f"y shape should be ({batch_size}, {block_size}), got {yb.shape}"
assert xb.dtype == torch.long, f"x should be LongTensor, got {xb.dtype}"
assert yb.dtype == torch.long, f"y should be LongTensor, got {yb.dtype}"

# Verify that y is x shifted by 1: for any batch element, y should start
# where x's second token is
# (This checks that the shift relationship holds)
assert torch.all(xb[:, 1:] == yb[:, :-1]), \
    "y should be x shifted right by 1 position"

# Check val split works too
xv, yv = get_batch('val')
assert xv.shape == (batch_size, block_size), "val batch x has wrong shape"

print(f"Train batch x shape: {xb.shape}")
print(f"Train batch y shape: {yb.shape}")
print(f"\nFirst training example (first 20 tokens):")
print(f"  x: {xb[0, :20].tolist()}")
print(f"  y: {yb[0, :20].tolist()}")
print(f"  x decoded: '{decode(xb[0, :20].tolist())}'")
print(f"  y decoded: '{decode(yb[0, :20].tolist())}'")
print("\nData batching sanity checks passed!")

class CharRNN_LM(nn.Module):
    """A character-level RNN language model using nn.RNN.

    Architecture:
        nn.Embedding(vocab_size, n_embd)
        -> nn.RNN(input_size=n_embd, hidden_size=n_embd, batch_first=True)
        -> nn.Linear(n_embd, vocab_size)

    The forward signature mirrors the TransformerLM you will build in Section 4:
    given idx of shape (B, T), return logits of shape (B, T, vocab_size).
    If targets are provided, also return the cross-entropy loss.
    """

    def __init__(self, vocab_size, n_embd):
        super().__init__()
        ## YOUR CODE HERE
        self.vocab_size = vocab_size
        self.token_emb = nn.Embedding(vocab_size, n_embd)
        self.rnn = nn.RNN(input_size=n_embd, hidden_size=n_embd, batch_first=True)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        """Forward pass.

        Args:
            idx: Token indices, shape (B, T).
            targets: Optional target indices, shape (B, T).

        Returns:
            logits: Shape (B, T, vocab_size).
            loss: Scalar cross-entropy loss if targets is provided, else None.

        Hint: Use F.cross_entropy on logits reshaped to (B*T, vocab_size) and
              targets reshaped to (B*T,).
        """
        ## YOUR CODE HERE
        B, T = idx.shape
        tok_emb = self.token_emb(idx)  # (B, T, n_embd)
        rnn_out, _ = self.rnn(tok_emb)  # (B, T, n_embd)
        logits = self.lm_head(rnn_out)  # (B, T, vocab_size)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(B*T, self.vocab_size), targets.view(B*T))

        return logits, loss

########## DO NOT MODIFY ##########

# Sanity check: forward pass produces logits of correct shape and a scalar loss
import torch.nn.functional as F
torch.manual_seed(0)
test_rnn = CharRNN_LM(vocab_size, n_embd).to(device)
xb_test, yb_test = get_batch('train')
logits_rnn, loss_rnn = test_rnn(xb_test, yb_test)

assert logits_rnn.shape == (batch_size, block_size, vocab_size), \
    f"Expected logits shape ({batch_size}, {block_size}, {vocab_size}), got {tuple(logits_rnn.shape)}"
assert loss_rnn is not None and loss_rnn.dim() == 0, "loss should be a scalar tensor when targets are provided"
print(f"CharRNN_LM forward OK. Logits shape: {tuple(logits_rnn.shape)}, initial loss: {loss_rnn.item():.4f}")
print(f"Random-baseline loss = ln(vocab_size) = {torch.log(torch.tensor(float(vocab_size))).item():.4f}")

# Skip when imported as a module (slow training loop)
if __name__ == '__main__':
    ########## DO NOT MODIFY ##########

    # Brief training of the RNN baseline for comparison with the Transformer (later in S4).
    # We run only 1000 iterations -- enough to see meaningful learning but not full training.

    torch.manual_seed(0)
    rnn_baseline = CharRNN_LM(vocab_size, n_embd).to(device)
    optimizer = torch.optim.AdamW(rnn_baseline.parameters(), lr=learning_rate)

    rnn_train_losses = []
    rnn_val_losses = []
    n_iters_baseline = 1000
    eval_interval_baseline = 100

    print("Training RNN baseline...")
    for step in range(n_iters_baseline):
        xb, yb = get_batch('train')
        _, loss = rnn_baseline(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step % eval_interval_baseline == 0 or step == n_iters_baseline - 1:
            with torch.no_grad():
                rnn_baseline.eval()
                xb_v, yb_v = get_batch('val')
                _, val_loss = rnn_baseline(xb_v, yb_v)
                rnn_baseline.train()
            rnn_train_losses.append(loss.item())
            rnn_val_losses.append(val_loss.item())
            print(f"  step {step:4d} | train loss {loss.item():.4f} | val loss {val_loss.item():.4f}")

    final_rnn_val_loss = rnn_val_losses[-1]
    print(f"\nFinal RNN baseline val loss after {n_iters_baseline} steps: {final_rnn_val_loss:.4f}")
    print("Keep this number in mind -- you'll compare it to the Transformer's val loss in Section 5.")

def get_causal_mask(seq_len):
    """Create a causal (lower-triangular) attention mask.
    
    Args:
        seq_len: Integer, the sequence length.
    
    Returns:
        A tensor of shape (seq_len, seq_len) where:
        - Lower triangle (including diagonal) contains 0.0
        - Upper triangle contains float('-inf')
    
    Example for seq_len=3:
        [[  0, -inf, -inf],
         [  0,    0, -inf],
         [  0,    0,    0]]
    """
    ## YOUR CODE HERE
    mask = torch.tril(torch.ones(seq_len, seq_len))
    mask = mask.masked_fill(mask == 0, float('-inf'))
    mask = mask.masked_fill(mask == 1, 0.0)
    return mask

# Sanity checks for causal mask

mask = get_causal_mask(4)

assert mask.shape == (4, 4), f"Mask shape should be (4, 4), got {mask.shape}"

# Check lower triangle is 0
for i in range(4):
    for j in range(i + 1):
        assert mask[i, j] == 0.0, f"Lower triangle at ({i},{j}) should be 0, got {mask[i,j]}"

# Check upper triangle is -inf
for i in range(4):
    for j in range(i + 1, 4):
        assert mask[i, j] == float('-inf'), \
            f"Upper triangle at ({i},{j}) should be -inf, got {mask[i,j]}"

# Check that applying softmax to masked scores gives zero for future positions
scores = torch.randn(4, 4).to(device)  # send to device for consistency
masked_scores = scores + mask.to(device)  # mask must be on same device as scores
attn_weights = F.softmax(masked_scores, dim=-1)
for i in range(4):
    for j in range(i + 1, 4):
        assert attn_weights[i, j].item() < 1e-6, \
            f"After softmax, future position ({i},{j}) should be ~0, got {attn_weights[i,j].item()}"

print("Causal mask:")
print(mask)
print(f"\nAfter softmax (row 2): {attn_weights[2].tolist()}")
print("  (Note: only positions 0, 1, 2 have nonzero weight)")
print("\nCausal mask sanity checks passed!")

class Head(nn.Module):
    """A single head of self-attention.
    
    Each head projects the input into Q, K, V of size head_size,
    computes scaled dot-product attention with causal masking,
    and returns the weighted combination of values.
    """
    
    def __init__(self, n_embd, head_size, dropout_rate):
        """Initialize a single attention head.
        
        Args:
            n_embd: Input embedding dimension.
            head_size: Dimension of Q, K, V for this head.
            dropout_rate: Dropout probability for attention weights.
        
        Create:
            self.query: nn.Linear(n_embd, head_size, bias=False)
            self.key:   nn.Linear(n_embd, head_size, bias=False)
            self.value: nn.Linear(n_embd, head_size, bias=False)
            self.dropout: nn.Dropout(dropout_rate)
        """
        super().__init__()
        ## YOUR CODE HERE
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.dropout = nn.Dropout(dropout_rate)
        self.attn_weights = None
    
    def forward(self, x):
        """Forward pass for a single attention head.
        
        Args:
            x: Input tensor of shape (B, T, n_embd)
               B = batch size, T = sequence length
        
        Returns:
            Output tensor of shape (B, T, head_size)
            self.attn_weights: Store the attention weights (after softmax,
                before dropout) as self.attn_weights for visualization later.
                Shape: (B, T, T)
        
        Steps:
            1. Compute Q, K, V by passing x through the linear layers
            2. Compute attention scores: Q @ K^T / sqrt(head_size)
            3. Apply causal mask (use get_causal_mask with seq_len = T)
           Device hint: get_causal_mask() returns a CPU tensor. Move it to
           the same device as scores with .to(scores.device) before adding,
           otherwise you will get a device mismatch error on CUDA.
            4. Apply softmax along the last dimension
            5. Store attention weights as self.attn_weights (for visualization)
            6. Apply dropout to attention weights
            7. Multiply attention weights by V
            8. Return the output
        """
        B, T, C = x.shape
        ## YOUR CODE HERE
        q = self.query(x)  # (B, T, head_size)
        k = self.key(x)    # (B, T, head_size)
        v = self.value(x)  # (B, T, head_size)

        scores = q @ k.transpose(-2, -1) / (q.shape[-1] ** 0.5)  # (B, T, T)

        mask = get_causal_mask(T).to(scores.device)
        scores = scores + mask

        attn = F.softmax(scores, dim=-1)  # (B, T, T)
        self.attn_weights = attn
        attn = self.dropout(attn)

        out = attn @ v  # (B, T, head_size)
        return out

# Sanity checks for Head

head_size_test = 32
test_head = Head(n_embd, head_size_test, dropout_rate)
test_head.eval()  # Turn off dropout for testing

# Check layer dimensions
assert test_head.query.in_features == n_embd, "Query input size should be n_embd"
assert test_head.query.out_features == head_size_test, "Query output size should be head_size"
assert test_head.key.in_features == n_embd, "Key input size should be n_embd"
assert test_head.value.in_features == n_embd, "Value input size should be n_embd"

# Check forward pass
B_test, T_test = 2, 8
x_test = torch.randn(B_test, T_test, n_embd)
with torch.no_grad():
    out_test = test_head(x_test)

assert out_test.shape == (B_test, T_test, head_size_test), \
    f"Output shape should be ({B_test}, {T_test}, {head_size_test}), got {out_test.shape}"

# Check that attention weights were stored and have correct shape
assert hasattr(test_head, 'attn_weights'), \
    "Head should store attention weights as self.attn_weights"
assert test_head.attn_weights.shape == (B_test, T_test, T_test), \
    f"Attention weights shape should be ({B_test}, {T_test}, {T_test}), got {test_head.attn_weights.shape}"

# Check that attention weights sum to 1 along last dimension
attn_sums = test_head.attn_weights.sum(dim=-1)
assert torch.allclose(attn_sums, torch.ones_like(attn_sums), atol=1e-5), \
    "Attention weights should sum to 1 along the last dimension"

# Check causality: upper triangle of attention weights should be ~0
for i in range(T_test):
    for j in range(i + 1, T_test):
        assert test_head.attn_weights[0, i, j].item() < 1e-6, \
            f"Position {i} should not attend to future position {j}"

print(f"Head output shape: {out_test.shape}")
print(f"Attention weights shape: {test_head.attn_weights.shape}")
print(f"Attention weights row 0 sums to: {test_head.attn_weights[0, 0].sum().item():.6f}")
print("\nSingle-head attention sanity checks passed!")

class MultiHeadAttention(nn.Module):
    """Multi-head self-attention: runs multiple Head modules in parallel.
    
    Each head operates on a slice of the embedding dimension.
    Outputs are concatenated and projected back to n_embd.
    """
    
    def __init__(self, n_embd, n_head, dropout_rate):
        """Initialize multi-head attention.
        
        Args:
            n_embd: Full embedding dimension.
            n_head: Number of attention heads.
            dropout_rate: Dropout probability.
        
        Create:
            self.heads: nn.ModuleList of n_head Head modules,
                        each with head_size = n_embd // n_head
            self.proj: nn.Linear(n_embd, n_embd) -- output projection
            self.dropout: nn.Dropout(dropout_rate)
        """
        super().__init__()
        ## YOUR CODE HERE
        head_size = n_embd // n_head
        self.heads = nn.ModuleList([Head(n_embd, head_size, dropout_rate) for _ in range(n_head)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x):
        """Forward pass for multi-head attention.
        
        Args:
            x: Input tensor of shape (B, T, n_embd)
        
        Returns:
            Output tensor of shape (B, T, n_embd)
        
        Steps:
            1. Run each head on x (each produces shape (B, T, head_size))
            2. Concatenate all head outputs along the last dimension using torch.cat(..., dim=-1)
               Result shape: (B, T, n_head * head_size) = (B, T, n_embd)
            3. Pass through the output projection linear layer
            4. Apply dropout
            5. Return the result
        """
        ## YOUR CODE HERE
        out = torch.cat([h(x) for h in self.heads], dim=-1)  # (B, T, n_embd)
        out = self.proj(out)
        out = self.dropout(out)
        return out

# Sanity checks for MultiHeadAttention

test_mha = MultiHeadAttention(n_embd, n_head, dropout_rate)
test_mha.eval()

# Check number of heads
assert len(test_mha.heads) == n_head, \
    f"Should have {n_head} heads, got {len(test_mha.heads)}"

# Check output projection
assert test_mha.proj.in_features == n_embd, "Projection input should be n_embd"
assert test_mha.proj.out_features == n_embd, "Projection output should be n_embd"

# Check forward pass: output shape should match input shape
B_test, T_test = 2, 8
x_test = torch.randn(B_test, T_test, n_embd)
with torch.no_grad():
    out_test = test_mha(x_test)

assert out_test.shape == (B_test, T_test, n_embd), \
    f"Output shape should be ({B_test}, {T_test}, {n_embd}), got {out_test.shape}"

# Each head should have its own attention weights
for i, head in enumerate(test_mha.heads):
    assert hasattr(head, 'attn_weights'), f"Head {i} should store attn_weights"

print(f"Number of heads: {len(test_mha.heads)}")
print(f"Head size: {n_embd // n_head}")
print(f"Input shape:  {x_test.shape}")
print(f"Output shape: {out_test.shape}")
print("\nMulti-head attention sanity checks passed!")

class FeedForward(nn.Module):
    """Position-wise feed-forward network.
    
    Two-layer MLP with ReLU activation and dropout.
    Inner dimension is 4 * n_embd.
    """
    
    def __init__(self, n_embd, dropout_rate):
        """Initialize the feed-forward network.
        
        Args:
            n_embd: Input/output dimension.
            dropout_rate: Dropout probability.
        
        Create self.net as an nn.Sequential with:
            1. nn.Linear(n_embd, 4 * n_embd)
            2. nn.ReLU()
            3. nn.Linear(4 * n_embd, n_embd)
            4. nn.Dropout(dropout_rate)
        """
        super().__init__()
        ## YOUR CODE HERE
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout_rate),
        )
    
    def forward(self, x):
        """Forward pass.
        
        Args:
            x: Tensor of shape (B, T, n_embd)
        
        Returns:
            Tensor of shape (B, T, n_embd)
        """
        ## YOUR CODE HERE
        return self.net(x)

# Sanity checks for FeedForward

test_ff = FeedForward(n_embd, dropout_rate)
test_ff.eval()

x_test = torch.randn(2, 8, n_embd)
with torch.no_grad():
    out_test = test_ff(x_test)

assert out_test.shape == x_test.shape, \
    f"FFN output shape {out_test.shape} should match input shape {x_test.shape}"

# Check that the inner dimension is 4 * n_embd
first_linear = test_ff.net[0]
assert first_linear.out_features == 4 * n_embd, \
    f"Inner dimension should be {4 * n_embd}, got {first_linear.out_features}"

print(f"FFN input shape:  {x_test.shape}")
print(f"FFN output shape: {out_test.shape}")
print(f"Inner dimension:  {first_linear.out_features}")
print("\nFeedForward sanity checks passed!")

class TransformerBlock(nn.Module):
    """A single transformer block: attention + feedforward with residual connections.
    
    Uses pre-norm architecture: LayerNorm is applied before each sublayer.
    """
    
    def __init__(self, n_embd, n_head, dropout_rate):
        """Initialize a transformer block.
        
        Args:
            n_embd: Embedding dimension.
            n_head: Number of attention heads.
            dropout_rate: Dropout probability.
        
        Create:
            self.ln1: nn.LayerNorm(n_embd) -- before attention
            self.attn: MultiHeadAttention(n_embd, n_head, dropout_rate)
            self.ln2: nn.LayerNorm(n_embd) -- before feedforward
            self.ff: FeedForward(n_embd, dropout_rate)
        """
        super().__init__()
        ## YOUR CODE HERE
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = MultiHeadAttention(n_embd, n_head, dropout_rate)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ff = FeedForward(n_embd, dropout_rate)
    
    def forward(self, x):
        """Forward pass with residual connections.
        
        Args:
            x: Input tensor of shape (B, T, n_embd)
        
        Returns:
            Output tensor of shape (B, T, n_embd)
        
        Computation (pre-norm with residual connections):
            x = x + self.attn(self.ln1(x))
            x = x + self.ff(self.ln2(x))
            return x
        """
        ## YOUR CODE HERE
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x

# Sanity checks for TransformerBlock

test_block = TransformerBlock(n_embd, n_head, dropout_rate)
test_block.eval()

B_test, T_test = 2, 8
x_test = torch.randn(B_test, T_test, n_embd)
with torch.no_grad():
    out_test = test_block(x_test)

assert out_test.shape == x_test.shape, \
    f"Block output shape {out_test.shape} should match input shape {x_test.shape}"

# Verify the block decomposes correctly (testing residual connections)
with torch.no_grad():
    ln1_out = test_block.ln1(x_test)
    attn_out = test_block.attn(ln1_out)
    after_attn = x_test + attn_out
    
    ln2_out = test_block.ln2(after_attn)
    ff_out = test_block.ff(ln2_out)
    expected = after_attn + ff_out

assert torch.allclose(out_test, expected, atol=1e-5), \
    "Block output should equal x + attn(ln1(x)) + ff(ln2(x + attn(ln1(x))))"

print(f"Block input shape:  {x_test.shape}")
print(f"Block output shape: {out_test.shape}")
print("Residual connection decomposition matches.")
print("\nTransformerBlock sanity checks passed!")

def test_residual_connections():
    """Test that residual connections pass through the input when sublayers output ~zero.
    
    Steps:
        1. Create a TransformerBlock and set to eval mode
        2. Zero out all parameters in self.attn and self.ff so their outputs are ~0
           (Hint: iterate over block.attn.parameters() and block.ff.parameters(),
            and set param.data to zeros using param.data.zero_())
        3. Create a random input x of shape (1, 8, n_embd)
        4. Run the block: output = block(x)
        5. Assert torch.allclose(output, x, atol=1e-4)
        6. Print a success message
    
    If this test fails, the residual connections are not implemented correctly.
    """
    ## YOUR CODE HERE
    block = TransformerBlock(n_embd, n_head, dropout_rate)
    block.eval()
    
    for param in block.attn.parameters():
        param.data.zero_()
    for param in block.ff.parameters():
        param.data.zero_()
    
    x = torch.randn(1, 8, n_embd)
    output = block(x)
    
    assert torch.allclose(output, x, atol=1e-4), \
        f"Residual connection test failed: output not close to input"
    print("Residual connection test passed!")

test_residual_connections()

class TransformerLM(nn.Module):
    """Decoder-only transformer language model.
    
    Combines token embeddings, position embeddings, transformer blocks,
    and a language model head for next-token prediction.
    """
    
    def __init__(self, vocab_size, n_embd, block_size, n_head, n_layer, dropout_rate):
        """Initialize the transformer language model.
        
        Args:
            vocab_size: Number of tokens in the vocabulary.
            n_embd: Embedding dimension.
            block_size: Maximum sequence length (context window).
            n_head: Number of attention heads per block.
            n_layer: Number of transformer blocks to stack.
            dropout_rate: Dropout probability.
        
        Create:
            self.token_emb: nn.Embedding(vocab_size, n_embd)
            self.pos_emb:   nn.Embedding(block_size, n_embd)
            self.blocks:    nn.Sequential of n_layer TransformerBlocks
            self.ln_f:      nn.LayerNorm(n_embd)  -- final layer norm
            self.lm_head:   nn.Linear(n_embd, vocab_size)  -- logit projection
        
        Also store:
            self.vocab_size = vocab_size  (for reference)
            self.block_size = block_size  (used by generate() to crop context)
        """
        super().__init__()
        ## YOUR CODE HERE
        self.vocab_size = vocab_size
        self.block_size = block_size
        
        self.token_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[
            TransformerBlock(n_embd, n_head, dropout_rate) for _ in range(n_layer)
        ])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
    
    def forward(self, idx, targets=None):
        """Forward pass: compute logits and optionally the loss.
        
        Args:
            idx: Token indices of shape (B, T), where B = batch size, T = seq length.
            targets: Optional target token indices of shape (B, T).
        
        Returns:
            logits: Tensor of shape (B, T, vocab_size) -- predicted token scores.
            loss: Cross-entropy loss (scalar) if targets provided, else None.
        
        Steps:
            1. Look up token embeddings: self.token_emb(idx)  -> (B, T, n_embd)
            2. Create position indices: torch.arange(T, device=idx.device)
            3. Look up position embeddings: self.pos_emb(positions)  -> (T, n_embd)
            4. Add token + position embeddings  -> (B, T, n_embd)
            5. Pass through self.blocks  -> (B, T, n_embd)
            6. Apply self.ln_f  -> (B, T, n_embd)
            7. Apply self.lm_head  -> (B, T, vocab_size)
            8. If targets is not None:
               - Reshape logits to (B*T, vocab_size) and targets to (B*T,)
               - Compute F.cross_entropy(logits, targets)
            9. Return logits (reshaped back if needed) and loss
        """
        B, T = idx.shape
        ## YOUR CODE HERE
        tok_emb = self.token_emb(idx)  # (B, T, n_embd)
        pos = torch.arange(T, device=idx.device)
        pos_emb = self.pos_emb(pos)  # (T, n_embd)
        x = tok_emb + pos_emb  # (B, T, n_embd)
        x = self.blocks(x)  # (B, T, n_embd)
        x = self.ln_f(x)  # (B, T, n_embd)
        logits = self.lm_head(x)  # (B, T, vocab_size)
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(B*T, self.vocab_size), targets.view(B*T))
        
        return logits, loss

# Sanity checks for TransformerLM

test_model = TransformerLM(vocab_size, n_embd, block_size, n_head, n_layer, dropout_rate).to(device)
test_model.eval()

# Check embedding layers
assert test_model.token_emb.num_embeddings == vocab_size, "Token embedding should have vocab_size entries"
assert test_model.token_emb.embedding_dim == n_embd, "Token embedding dim should be n_embd"
assert test_model.pos_emb.num_embeddings == block_size, "Position embedding should have block_size entries"
assert test_model.pos_emb.embedding_dim == n_embd, "Position embedding dim should be n_embd"

# Check forward pass without targets
B_test, T_test = 2, 16
idx_test = torch.randint(0, vocab_size, (B_test, T_test), device=device)
with torch.no_grad():
    logits_test, loss_test = test_model(idx_test)

assert logits_test.shape == (B_test, T_test, vocab_size), \
    f"Logits shape should be ({B_test}, {T_test}, {vocab_size}), got {logits_test.shape}"
assert loss_test is None, "Loss should be None when targets are not provided"

# Check forward pass with targets
targets_test = torch.randint(0, vocab_size, (B_test, T_test), device=device)
with torch.no_grad():
    logits_test2, loss_test2 = test_model(idx_test, targets_test)

assert loss_test2 is not None, "Loss should not be None when targets are provided"
assert loss_test2.dim() == 0, f"Loss should be a scalar, got shape {loss_test2.shape}"
# With random weights, loss should be approximately -log(1/vocab_size) = log(vocab_size)
expected_loss = np.log(vocab_size)
assert abs(loss_test2.item() - expected_loss) < 1.0, \
    f"Initial loss should be near {expected_loss:.2f} (random guessing), got {loss_test2.item():.2f}"

# Count parameters
n_params = sum(p.numel() for p in test_model.parameters())
print(f"TransformerLM architecture:")
print(f"  vocab_size = {vocab_size}")
print(f"  n_embd     = {n_embd}")
print(f"  block_size = {block_size}")
print(f"  n_head     = {n_head}")
print(f"  n_layer    = {n_layer}")
print(f"  Parameters = {n_params:,}")
print(f"\nLogits shape (no targets): {logits_test.shape}")
print(f"Loss (with targets):       {loss_test2.item():.4f}")
print(f"Expected initial loss:     ~{expected_loss:.4f} (random guessing)")
print("\nTransformerLM sanity checks passed!")

def train_model(model, n_iters=5000, eval_interval=500):
    """Train the transformer language model.

    Args:
        model: A TransformerLM model (already on the correct device).
        n_iters: Number of training iterations.
        eval_interval: Print train/val loss every this many steps.

    Returns:
        train_losses: List of training losses recorded at each eval_interval.
        val_losses: List of validation losses recorded at each eval_interval.

    Steps:
        1. Create an AdamW optimizer: `torch.optim.AdamW(model.parameters(), lr=learning_rate)`
           (uses the global `learning_rate` from Section 0.2)
        2. For each iteration:
           a. Sample a batch from the training set using get_batch('train').
           b. Forward pass: compute logits and loss.
           c. `optimizer.zero_grad(set_to_none=True)` -> `loss.backward()` -> `optimizer.step()`
           d. Every eval_interval steps:
              - Evaluate on a validation batch (with torch.no_grad())
              - Record `loss.item()` (a float) to train_losses / val_losses
              - Print step, train loss, and val loss
    """
    ## YOUR CODE HERE
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    train_losses = []
    val_losses = []
    
    for step in range(n_iters):
        xb, yb = get_batch('train')
        logits, loss = model(xb, yb)
        
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        
        if step % eval_interval == 0 or step == n_iters - 1:
            with torch.no_grad():
                model.eval()
                xb_v, yb_v = get_batch('val')
                _, val_loss = model(xb_v, yb_v)
                model.train()
            train_losses.append(loss.item())
            val_losses.append(val_loss.item())
            print(f"  step {step:5d} | train loss {loss.item():.4f} | val loss {val_loss.item():.4f}")
    
    return train_losses, val_losses

# Skip when imported as a module (slow training loop)
if __name__ == '__main__':
    # Create and train the model
    model = TransformerLM(vocab_size, n_embd, block_size, n_head, n_layer, dropout_rate).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")
    print(f"Training for 2500 iterations...\n")

    start_time = time.time()
    train_losses, val_losses = train_model(model, n_iters=2500, eval_interval=250)
    elapsed = time.time() - start_time

    print(f"\nTraining complete in {elapsed / 60:.2f} minutes.")
    print(f"Final train loss: {train_losses[-1]:.4f}")
    print(f"Final val loss:   {val_losses[-1]:.4f}")

# Skip when imported as a module (needs trained model)
if __name__ == '__main__':
    # Plot training and validation loss
    plt.figure(figsize=(10, 4))
    steps = [i * 250 for i in range(len(train_losses))]
    # Use .item() in case losses were stored as tensors rather than floats
    train_vals = [l.item() if hasattr(l, 'item') else l for l in train_losses]
    val_vals   = [l.item() if hasattr(l, 'item') else l for l in val_losses]
    plt.plot(steps, train_vals, label='Train Loss', alpha=0.8)
    plt.plot(steps, val_vals,   label='Val Loss', alpha=0.8)
    plt.xlabel('Training Iterations')
    plt.ylabel('Cross-Entropy Loss')
    plt.title('Transformer Language Model Training')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def generate(model, prompt, max_new_tokens=500, temperature=1.0):
    """Generate text autoregressively from a prompt.
    
    Args:
        model: A trained TransformerLM model.
        prompt: A string to use as the starting context.
        max_new_tokens: Number of new tokens to generate.
        temperature: Sampling temperature (default 1.0).
    
    Returns:
        A string containing the prompt followed by the generated text.
    
    Steps:
        1. Encode the prompt to a tensor of shape (1, len(prompt)) on the correct device.
           Use: torch.tensor([encode(prompt)], device=device)
        2. Set model to eval mode and use torch.no_grad().
        3. For each new token:
           a. Crop: `idx_cond = idx[:, -model.block_size:]`  (keeps the last block_size tokens)
           b. Forward pass to get logits of shape (1, T, vocab_size).
           c. Take logits for the last position only: logits[:, -1, :]
           d. Divide logits by temperature.
           e. Apply softmax to get probabilities.
           f. Sample next token using torch.multinomial(probs, num_samples=1).
           g. Append the sampled token to the context.
        4. Decode the full sequence (prompt + generated) back to a string.
    """
    ## YOUR CODE HERE
    idx = torch.tensor([encode(prompt)], device=device)
    
    model.eval()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -model.block_size:]
            logits, _ = model(idx_cond)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, idx_next], dim=1)
    
    return decode(idx[0].tolist())

# Skip when imported as a module (needs trained model)
if __name__ == '__main__':
    # Sanity check: generate should produce a string
    test_output = generate(model, "ROMEO:", max_new_tokens=10, temperature=1.0)
    assert isinstance(test_output, str), f"generate should return a string, got {type(test_output)}"
    assert test_output.startswith("ROMEO:"), "Output should start with the prompt"
    assert len(test_output) > len("ROMEO:"), "Output should be longer than the prompt"

    print("Generation sanity check passed!\n")
    print("="*60)
    print("Generated text (temperature=1.0):")
    print("="*60)
    print(generate(model, "ROMEO:\nWherefore art thou?", max_new_tokens=300, temperature=1.0))

# Skip when imported as a module (needs trained model)
if __name__ == '__main__':
    # Compare different temperatures
    prompt = "To be, or not to be,"

    for temp in [0.5, 1.0, 1.5]:
        print(f"\n{'='*60}")
        print(f"Temperature = {temp}")
        print(f"{'='*60}")
        print(generate(model, prompt, max_new_tokens=200, temperature=temp))

# Skip when imported as a module (needs trained model)
if __name__ == '__main__':
    # Use this cell to extract and visualize attention patterns.

    sample_text = "To be, or not to be,"
    sample_tokens = torch.tensor([encode(sample_text)], device=device)

    # Forward pass to populate attention weights
    model.eval()
    with torch.no_grad():
        _ = model(sample_tokens)

    # Example: access attention weights from layer 0, head 0
    attn_weights_00 = model.blocks[0].attn.heads[0].attn_weights[0].cpu().numpy()
    # Shape: (T, T) where T = len(sample_text)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(attn_weights_00, cmap='viridis', aspect='auto')
    ax.set_xticks(range(len(sample_text))); ax.set_xticklabels(list(sample_text), fontsize=8)
    ax.set_yticks(range(len(sample_text))); ax.set_yticklabels(list(sample_text), fontsize=8)
    ax.set_title('Layer 0, Head 0'); plt.colorbar(im, ax=ax); plt.tight_layout(); plt.show()
    print("Add 3 more head visualizations below:")

    ## YOUR CODE HERE
    # Plot at least 4 different heads total (across at least 2 layers).
    # The example below plots layer 0, head 0 -- complete 3 more on your own.
    # 1. Extract attention weights from at least 2 heads
    # 2. Plot heatmaps using plt.imshow
    #    - Use the characters of sample_text as tick labels
    #    - Add a colorbar
    #    - Label which layer/head each plot shows

# Additional attention visualizations
if __name__ == '__main__':
    # Layer 0, Head 2
    fig, ax = plt.subplots(figsize=(6, 5))
    attn = model.blocks[0].attn.heads[2].attn_weights[0].cpu().numpy()
    im = ax.imshow(attn, cmap='viridis', aspect='auto')
    ax.set_xticks(range(len(sample_text))); ax.set_xticklabels(list(sample_text), fontsize=8)
    ax.set_yticks(range(len(sample_text))); ax.set_yticklabels(list(sample_text), fontsize=8)
    ax.set_title('Layer 0, Head 2'); plt.colorbar(im, ax=ax); plt.tight_layout(); plt.show()

    # Layer 2, Head 0
    fig, ax = plt.subplots(figsize=(6, 5))
    attn = model.blocks[2].attn.heads[0].attn_weights[0].cpu().numpy()
    im = ax.imshow(attn, cmap='viridis', aspect='auto')
    ax.set_xticks(range(len(sample_text))); ax.set_xticklabels(list(sample_text), fontsize=8)
    ax.set_yticks(range(len(sample_text))); ax.set_yticklabels(list(sample_text), fontsize=8)
    ax.set_title('Layer 2, Head 0'); plt.colorbar(im, ax=ax); plt.tight_layout(); plt.show()

    # Layer 3, Head 1
    fig, ax = plt.subplots(figsize=(6, 5))
    attn = model.blocks[3].attn.heads[1].attn_weights[0].cpu().numpy()
    im = ax.imshow(attn, cmap='viridis', aspect='auto')
    ax.set_xticks(range(len(sample_text))); ax.set_xticklabels(list(sample_text), fontsize=8)
    ax.set_yticks(range(len(sample_text))); ax.set_yticklabels(list(sample_text), fontsize=8)
    ax.set_title('Layer 3, Head 1'); plt.colorbar(im, ax=ax); plt.tight_layout(); plt.show()
