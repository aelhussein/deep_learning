##import functions
import torch
import torch.nn as nn
from torch.nn import functional as F
import math

##hyperparameters
batch_size = 64
block_size = 256
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2
# ----------------

torch.manual_seed(1337)

## load data file
input = '/Users/ae2772/Documents/Moocs/mini-GPT3/tiny_shakespear.txt'
with open(f'{input}', 'r', encoding = 'utf-8') as f:
    text = f.read()

##get all unique characters in text
chars = sorted(list(set(text)))
vocab_size = len(chars)


##create mapping from characters to integers to enable encoding and decoding
stoi = {ch:i for i,ch in enumerate(chars)} # each char associated with value
itos = {i:ch for i,ch in enumerate(chars)} # each value associated with car
encode = lambda s: [stoi[c] for c in s] ## encoder - consider using SentencePiece for sub-word tokenizer (more effective)
decode = lambda l: ''.join([itos[i] for i in l]) ## decoder

## encode the test and store as tensor
data = torch.tensor(encode(text), dtype = torch.long)

## split data in train and test
n = int(0.9*len(data))
train_data = data[:n]
test_data = data[n:]

## data loading
def get_batch(split):
    ## generate batch of inputs and targets
    data = train_data if split == 'train' else test_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

def new_gelu(x): ## implement new gelu activation
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

@torch.no_grad() ## never call backward, so reduce memory overhead
def estimate_loss():
    ## average losses across many batches
    out = {}
    model.eval() ## create evaluation mode of model
    for split in ['train', 'test']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X,Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train() ## reset back to train mode
    return out

class Head(nn.Module):
    ## one slef-attention head
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self,x):
        B,T,C = x.shape
        k = self.key(x) # (B, T, C)
        q = self.query(x) # (B, T, C)
        ##Compute attention scores
        weights = q @ k.transpose(-2, -1) * C **-0.5 # (B,T,C) @ (B,C, T) = (B,T,T)
        weights = weights.masked_fill(self.tril[:T,:T] == 0, float('-inf')) ## all elements where trill = 0 make inf
        weights = F.softmax(weights, dim = -1) ## normalization operation
        weights = self.dropout(weights)
        v = self.value(x)
        out = weights @ v
        return out
            
class MultiHeadAttention(nn.Module):
    ##multiple self-attention heads in parallel
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self,x):
        out  = torch.cat([h(x) for h in self.heads], dim = -1)
        out = self.proj(out)
        out = self.dropout(out)
        return out

class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
        )
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        x = self.net(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):
    ## transformer block
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.ln1 = nn.LayerNorm(n_embd)
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ffwd = FeedForward(n_embd)
        
    def forward(self, x):
        x = x + self.sa(self.ln1(x)) ## add residual connections i.e. x + ...
        x = x + self.ffwd(self.ln2(x))
        return x

## simple bigram model that doesnt take into account much context
class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__() ## load from nn.Module initializations
        ## each token reads off the logits from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd) 
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential( * [Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) ## final norm layer
        self.lm_head = nn.Linear(n_embd, vocab_size) ## turn embedding back to logits

    def forward(self, idx, targets = None):
        B,T = idx.shape

        tok_emb =  self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device = device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x) ## arrange by (batch, time/context, vocab_size)
        
        if targets is None:
            ## in case where want to generate new text without a target
            loss = None
        else:
            ## reshape tensor to meet cross_entropy loss
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets) ## measure the quality of the logits vs. target

        return logits, loss

    def generate(self, idx, max_new_tokens):
        ## generate new text based on logits
        ## idx = (B,T) array in the current context
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond) ## get predictions using forward pass
            logits = logits[:,-1,:] ## get the last time step i.e. (B, C)
            probs = F.softmax(logits, dim = 1) ## apply softmax to get probs
            idx_next = torch.multinomial(probs, num_samples = 1) ## sample from multinomial distribution i.e (B, 1)
            idx = torch.cat((idx, idx_next), dim = 1) ## append sampled to running sequence i.e. (B, T+1)
        return idx

model = BigramLanguageModel()
#m  = model.to(device)
## create optimizer (Adam)
optimizer = torch.optim.AdamW(model.parameters(), lr = learning_rate)

## train model
for iter in range(max_iters):
    if iter % eval_interval == 0 or iter == max_iters -1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, test loss {losses['test']:.4f}")
    
    for i in range(1000):
        ##sample batches
        xb, yb = get_batch('train')
        
        ## evaluate loss
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
    optimizer.step()

context = torch.zeros((1,1), dtype = torch.long, device = device) ## initialize first character to 0
print(decode(model.generate(context, max_new_tokens = 500)[0].tolist()))
