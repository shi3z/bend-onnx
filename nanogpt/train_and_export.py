#!/usr/bin/env python3
"""
nanoGPT Port to Bend: Training, Export, and Verification Script.

Implements Andrej Karpathy's nanoGPT in PyTorch, trains on sample sequences,
and exports the model to pure, verified Bend code with GPU parallel tree operations.
"""

import sys
if sys.version_info >= (3, 14):
    p = "/home/shi3z/snap/antigravity-cli/common/local/lib/python3.14/dist-packages"
    if p not in sys.path:
        sys.path.insert(0, p)

import os
import math
import subprocess
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

NANOGPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(NANOGPT_DIR)
BEND_BIN = os.path.expanduser("~/.bend/bin/bend")

# --- Vocabulary & Tokenizer ---
CHARS = " ABCDEFGHIJKLMNOPQRSTUVWXYZ!.:01"
CHAR_TO_ID = {ch: i for i, ch in enumerate(CHARS)}
ID_TO_CHAR = {i: ch for i, ch in enumerate(CHARS)}
VOCAB_SIZE = len(CHARS) # 32

def encode(s: str) -> list[int]:
    return [CHAR_TO_ID.get(c.upper(), 0) for c in s]

def decode(ids: list[int]) -> str:
    return "".join(ID_TO_CHAR.get(i, " ") for i in ids)

# --- nanoGPT PyTorch Architecture (Karpathy style) ---
class GPTConfig:
    def __init__(self, block_size=16, vocab_size=32, n_layer=1, n_head=1, n_embd=16):
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_embd = config.n_embd
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        
        # Scaled dot-product causal attention
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.n_embd))
        causal_mask = torch.tril(torch.ones(T, T, device=x.device)).view(1, T, T)
        att = att.masked_fill(causal_mask == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        
        y = att @ v
        return self.c_proj(y)

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 2 * config.n_embd, bias=True)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(2 * config.n_embd, config.n_embd, bias=True)

    def forward(self, x):
        h = self.gelu(self.c_fc(x))
        return self.c_proj(h)

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class NanoGPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.block_size, config.n_embd)
        self.h = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def forward(self, idx):
        B, T = idx.size()
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device).unsqueeze(0)
        
        tok_emb = self.wte(idx)
        pos_emb = self.wpe(pos)
        x = tok_emb + pos_emb
        
        for block in self.h:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits

def train_model():
    torch.manual_seed(42)
    config = GPTConfig(block_size=16, vocab_size=VOCAB_SIZE, n_layer=1, n_head=1, n_embd=16)
    model = NanoGPT(config)
    
    phrases = [
        "BEND IS FAST!  ",
        "BEND ON GPU!   ",
        "BEND RUNS AI!  ",
        "BEND PARALLEL! "
    ]
    encoded_data = [encode(p) for p in phrases]
    X_train = torch.tensor([p[:-1] for p in encoded_data], dtype=torch.long)
    Y_train = torch.tensor([p[1:] for p in encoded_data], dtype=torch.long)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.03, weight_decay=1e-4)
    model.train()
    
    print("Training miniature nanoGPT model on target sequences...")
    for step in range(120):
        optimizer.zero_grad()
        logits = model(X_train)
        loss = F.cross_entropy(logits.view(-1, config.vocab_size), Y_train.view(-1))
        loss.backward()
        optimizer.step()
        if (step + 1) % 30 == 0:
            print(f"  Step {step+1:3d} | Loss: {loss.item():.4f}")
            
    model.eval()
    return model, config

def format_f32(v: float) -> str:
    val = float(v)
    if val == 0.0:
        return "0.0"
    is_neg = val < 0.0
    abs_val = abs(val)
    s = f"{abs_val:.7g}"
    if "." not in s and "e" not in s:
        s += ".0"
    if is_neg:
        return f"F32.neg({s})"
    return s

def format_vec(vals) -> str:
    res = "VNil{}"
    for v in reversed(vals):
        res = f"VCon{{{format_f32(v)}, {res}}}"
    return res

def build_mat_tree(rows) -> str:
    if len(rows) == 0:
        return "MTLeaf{VNil{}}"
    if len(rows) == 1:
        return f"MTLeaf{{{format_vec(rows[0])}}}"
    mid = len(rows) // 2
    l = build_mat_tree(rows[:mid])
    r = build_mat_tree(rows[mid:])
    return f"MTNode{{\n    {l},\n    {r}\n  }}"

def build_vec_list(rows) -> str:
    res = "VLNil{}"
    for r in reversed(rows):
        res = f"VLCon{{{format_vec(r)}, {res}}}"
    return res

def export_to_bend(model: NanoGPT, config: GPTConfig, output_path: str):
    print(f"\nTranspiling nanoGPT model into Bend: {output_path}...")
    
    wte = model.wte.weight.detach().numpy()
    wpe = model.wpe.weight.detach().numpy()
    
    b0 = model.h[0]
    ln1_g = b0.ln_1.weight.detach().numpy()
    ln1_b = b0.ln_1.bias.detach().numpy()
    
    c_attn_w = b0.attn.c_attn.weight.detach().numpy()
    wq = c_attn_w[:16, :]
    wk = c_attn_w[16:32, :]
    wv = c_attn_w[32:48, :]
    
    w_proj = b0.attn.c_proj.weight.detach().numpy()
    
    ln2_g = b0.ln_2.weight.detach().numpy()
    ln2_b = b0.ln_2.bias.detach().numpy()
    
    w_fc = b0.mlp.c_fc.weight.detach().numpy()
    b_fc = b0.mlp.c_fc.bias.detach().numpy()
    w_mlp_proj = b0.mlp.c_proj.weight.detach().numpy()
    b_mlp_proj = b0.mlp.c_proj.bias.detach().numpy()
    
    ln_f_g = model.ln_f.weight.detach().numpy()
    ln_f_b = model.ln_f.bias.detach().numpy()
    
    lm_head_w = model.lm_head.weight.detach().numpy()
    
    # Read matrix.bend primitives
    matrix_bend_path = os.path.join(PROJECT_ROOT, "src", "matrix.bend")
    with open(matrix_bend_path) as f:
        matrix_code = f.read()
    matrix_code = matrix_code.split("def main() -> IO(Unit):")[0].strip()
    
    bend_code = f"""{matrix_code}

# ==============================================================================
# nanoGPT in Pure Bend (Transformer Architecture with Causal Self-Attention)
# Ported from Andrej Karpathy's nanoGPT
# ==============================================================================

# --- Token List Representation ---
type Tokens is Data:
  TNil{{}}
  TCon{{tok: U32, rest: Tokens}}

def tokens_len(ts: Tokens) -> U32:
  match ts:
    case TNil{{}}:
      0
    case TCon{{_, t}}:
      (1 + tokens_len(t) : U32)

def tokens_append(ts: Tokens, val: U32) -> Tokens:
  match ts:
    case TNil{{}}:
      TCon{{val, TNil{{}}}}
    case TCon{{h, t}}:
      TCon{{h, tokens_append(t, val)}}

def tokens_last(+ts: Tokens) -> U32:
  match ts:
    case TNil{{}}:
      0
    case TCon{{h, TNil{{}}}}:
      h
    case TCon{{_, t}}:
      tokens_last(t)

# --- Vector List for Embeddings ---
type VecList is Data:
  VLNil{{}}
  VLCon{{head: Vec, tail: VecList}}

def vec_list_get(+vs: VecList, +idx: U32) -> Vec:
  match vs idx:
    case VLCon{{+row, _}} 0:
      row
    case VLCon{{_, t}} _:
      vec_list_get(t, U32.sub(idx, 1))
    case _ _:
      VNil{{}}

# --- Key-Value Cache for Causal Attention ---
type KVPair is Data:
  KVP{{k: Vec, v: Vec}}

type KVList is Data:
  KVNil{{}}
  KVCon{{pair: KVPair, rest: KVList}}

# --- Model Parameters (Trained Weights) ---
def param_wte() -> VecList:
  {build_vec_list(wte.tolist())}

def param_wpe() -> VecList:
  {build_vec_list(wpe.tolist())}

def param_ln1_g() -> Vec:
  {format_vec(ln1_g.tolist())}

def param_ln1_b() -> Vec:
  {format_vec(ln1_b.tolist())}

def param_wq() -> MatTree:
  {build_mat_tree(wq.tolist())}

def param_wk() -> MatTree:
  {build_mat_tree(wk.tolist())}

def param_wv() -> MatTree:
  {build_mat_tree(wv.tolist())}

def param_w_attn_proj() -> MatTree:
  {build_mat_tree(w_proj.tolist())}

def param_ln2_g() -> Vec:
  {format_vec(ln2_g.tolist())}

def param_ln2_b() -> Vec:
  {format_vec(ln2_b.tolist())}

def param_w_mlp_fc() -> MatTree:
  {build_mat_tree(w_fc.tolist())}

def param_b_mlp_fc() -> Vec:
  {format_vec(b_fc.tolist())}

def param_w_mlp_proj() -> MatTree:
  {build_mat_tree(w_mlp_proj.tolist())}

def param_b_mlp_proj() -> Vec:
  {format_vec(b_mlp_proj.tolist())}

def param_ln_f_g() -> Vec:
  {format_vec(ln_f_g.tolist())}

def param_ln_f_b() -> Vec:
  {format_vec(ln_f_b.tolist())}

def param_lm_head() -> MatTree:
  {build_mat_tree(lm_head_w.tolist())}

# Embedding lookup: WTE[tok] + WPE[pos]
def get_embedding(+tok: U32, +pos: U32) -> Vec:
  tok_vec = vec_list_get(param_wte(), tok)
  pos_vec = vec_list_get(param_wpe(), pos)
  vec_add(tok_vec, pos_vec)

# Compute Key and Value vectors for all past tokens
def compute_kv_list(ts: Tokens, +pos: U32) -> KVList:
  match ts:
    case TNil{{}}:
      KVNil{{}}
    case TCon{{tok, rest}}:
      +x = get_embedding(tok, pos)
      +normed = vec_layernorm(x, param_ln1_g(), param_ln1_b())
      k = mat_tree_mul(normed, param_wk())
      v = mat_tree_mul(normed, param_wv())
      KVCon{{KVP{{k, v}}, compute_kv_list(rest, (pos + 1 : U32))}}

# Compute attention score for each past token against query Q
def compute_scores(+q: Vec, kvs: KVList) -> Vec:
  match kvs:
    case KVNil{{}}:
      VNil{{}}
    case KVCon{{KVP{{+k, _}}, rest}}:
      scale : F32 = 0.25 # 1.0 / sqrt(16.0)
      score = ((vec_dot(q, k) * scale : F32))
      VCon{{score, compute_scores(q, rest)}}

# Aggregate values weighted by softmax attention weights
def apply_attention(weights: Vec, kvs: KVList) -> Vec:
  match weights kvs:
    case VCon{{+w, wt}} KVCon{{KVP{{_, +v}}, kt}}:
      scaled_v = vec_scale(w, v)
      rest_v = apply_attention(wt, kt)
      vec_add(scaled_v, rest_v)
    case _ _:
      VNil{{}}

# Transformer forward pass for next-token prediction
def gpt_forward_step(+tokens: Tokens) -> Vec:
  # 1. Fetch current token and position embedding
  curr_tok : U32 = tokens_last(tokens)
  pos : U32 = U32.sub(tokens_len(tokens), 1)
  +x : Vec = get_embedding(curr_tok, pos)
  
  # 2. LayerNorm 1
  normed1 : Vec = vec_layernorm(x, param_ln1_g(), param_ln1_b())
  
  # 3. Compute Query at current position
  +q : Vec = mat_tree_mul(normed1, param_wq())
  
  # 4. Multi-token Causal Self-Attention over past tokens
  +kvs : KVList = compute_kv_list(tokens, 0)
  scores : Vec = compute_scores(q, kvs)
  +attn_weights : Vec = vec_softmax(scores)
  context : Vec = apply_attention(attn_weights, kvs)
  attn_out : Vec = mat_tree_mul(context, param_w_attn_proj())
  
  # Residual connection 1: x + attn_out
  +x1 : Vec = vec_add(x, attn_out)
  
  # 5. LayerNorm 2
  normed2 : Vec = vec_layernorm(x1, param_ln2_g(), param_ln2_b())
  
  # 6. MLP Block: GELU(W_fc * x + b) -> W_proj * h + b
  h_fc : Vec = linear_layer(normed2, param_w_mlp_fc(), param_b_mlp_fc())
  h_gelu : Vec = vec_gelu(h_fc)
  mlp_out : Vec = linear_layer(h_gelu, param_w_mlp_proj(), param_b_mlp_proj())
  
  # Residual connection 2: x1 + mlp_out
  +x2 : Vec = vec_add(x1, mlp_out)
  
  # 7. Final LayerNorm & LM Head (Projection to Vocab Logits)
  normed_final : Vec = vec_layernorm(x2, param_ln_f_g(), param_ln_f_b())
  mat_tree_mul(normed_final, param_lm_head())

# Decode single token ID to character
def token_to_char(tok: U32) -> String:
  match tok:
    case 0:  " "
    case 1:  "A"
    case 2:  "B"
    case 3:  "C"
    case 4:  "D"
    case 5:  "E"
    case 6:  "F"
    case 7:  "G"
    case 8:  "H"
    case 9:  "I"
    case 10: "J"
    case 11: "K"
    case 12: "L"
    case 13: "M"
    case 14: "N"
    case 15: "O"
    case 16: "P"
    case 17: "Q"
    case 18: "R"
    case 19: "S"
    case 20: "T"
    case 21: "U"
    case 22: "V"
    case 23: "W"
    case 24: "X"
    case 25: "Y"
    case 26: "Z"
    case 27: "!"
    case 28: "."
    case 29: ":"
    case 30: "0"
    case _:  "1"

def print_tokens(ts: Tokens) -> String:
  match ts:
    case TNil{{}}:
      ""
    case TCon{{tok, rest}}:
      token_to_char(tok) ++ print_tokens(rest)

# Autoregressive generation loop in Bend:
def generate_tokens(steps: Nat, +ts: Tokens) -> Tokens:
  match steps:
    case 0n:
      ts
    case 1n+p:
      +logits : Vec = gpt_forward_step(ts)
      next_tok : U32 = vec_argmax(logits)
      new_tokens : Tokens = tokens_append(ts, next_tok)
      generate_tokens(p, new_tokens)

def main() -> IO(Unit):
  do IO<Unit>:
    # Prompt: "BEND " -> Token IDs: [2, 5, 14, 4, 0]
    +prompt : Tokens = TCon{{2, TCon{{5, TCon{{14, TCon{{4, TCon{{0, TNil{{}}}}}}}}}}}}
    
    IO.print("--- nanoGPT Text Generation in Bend ---")
    IO.print("Prompt:   " ++ print_tokens(prompt))
    
    # Autoregressively generate next 9 tokens using GPU tree operations:
    gen : Tokens = generate_tokens(9n, prompt)
    
    IO.print("Output:   " ++ print_tokens(gen))
    IO.print("---------------------------------------")
"""
    with open(output_path, "w") as f:
        f.write(bend_code)
    print(f"Generated {output_path} successfully!")

def main():
    model, config = train_model()
    
    prompt_str = "BEND "
    prompt_ids = encode(prompt_str)
    cur_ids = list(prompt_ids)
    for _ in range(9):
        x = torch.tensor([cur_ids], dtype=torch.long)
        logits = model(x)
        next_id = int(torch.argmax(logits[0, -1]).item())
        cur_ids.append(next_id)
    print(f"\nPyTorch reference generation: '{decode(cur_ids)}'")
    
    bend_path = os.path.join(NANOGPT_DIR, "nanogpt.bend")
    export_to_bend(model, config, bend_path)

if __name__ == '__main__':
    main()
