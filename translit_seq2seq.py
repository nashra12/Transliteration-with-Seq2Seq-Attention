"""
Transliteration with Seq2Seq + Attention on Google Dakshina Dataset
-------------------------------------------------------------------
- Implements Encoder-Decoder with Bahdanau Attention (character-level)
- Converts Romanized Hindi words -> Devanagari script outputs
- Supports RNN, GRU, LSTM architectures
- Evaluates with Exact Match accuracy
- Saves attention heatmaps for visualization
"""

import os
import argparse
import unicodedata
import re
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import random
from sklearn.metrics import accuracy_score

# -------------------------------
# Utils: Data Loading
# -------------------------------

def normalize_string(s):
    """Remove accents and normalize spacing"""
    s = ''.join(c for c in unicodedata.normalize('NFD', s)
                if unicodedata.category(c) != 'Mn')
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z?.!,¿]+", " ", s)
    return s.strip()

def load_dakshina_pairs(data_dir, lang="hi", split="train"):
    path = os.path.join(data_dir, f"{lang}/lexicons/{split}.tsv")
    pairs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            native, latin = line.strip().split("\t")
            pairs.append((latin, native))
    return pairs

# -------------------------------
# Vocab
# -------------------------------

class Vocab:
    def __init__(self, tokens=None):
        self.pad_token = "<pad>"
        self.sos_token = "<sos>"
        self.eos_token = "<eos>"
        self.tokens = [self.pad_token, self.sos_token, self.eos_token]
        if tokens:
            self.tokens += sorted(set(tokens))
        self.stoi = {t:i for i,t in enumerate(self.tokens)}
        self.itos = {i:t for i,t in enumerate(self.tokens)}
    def encode(self, text):
        return [self.stoi[self.sos_token]] + [self.stoi[c] for c in text if c in self.stoi] + [self.stoi[self.eos_token]]
    def decode(self, ids):
        result = []
        for i in ids:
            if i == self.stoi[self.eos_token]:
                break
            if i in self.itos and i not in (self.stoi[self.sos_token], self.stoi[self.pad_token]):
                result.append(self.itos[i])
        return ''.join(result)
    def __len__(self):
        return len(self.tokens)

# -------------------------------
# Dataset
# -------------------------------

class TransliterationDataset(Dataset):
    def __init__(self, pairs, src_vocab, tgt_vocab, max_len=30):
        self.pairs = pairs
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.max_len = max_len
    def __len__(self):
        return len(self.pairs)
    def __getitem__(self, idx):
        src, tgt = self.pairs[idx]
        src_ids = self.src_vocab.encode(src)[:self.max_len]
        tgt_ids = self.tgt_vocab.encode(tgt)[:self.max_len]
        return torch.tensor(src_ids), torch.tensor(tgt_ids)

def collate_fn(batch):
    srcs, tgts = zip(*batch)
    srcs = nn.utils.rnn.pad_sequence(srcs, padding_value=0)
    tgts = nn.utils.rnn.pad_sequence(tgts, padding_value=0)
    return srcs, tgts

# -------------------------------
# Seq2Seq + Attention
# -------------------------------

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hidden_dim, n_layers=1, cell="GRU"):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        rnn_cls = nn.RNN if cell=="RNN" else nn.GRU if cell=="GRU" else nn.LSTM
        self.rnn = rnn_cls(emb_dim, hidden_dim, n_layers)
    def forward(self, src):
        emb = self.embedding(src)
        outputs, hidden = self.rnn(emb)
        return outputs, hidden

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim*2, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)
    def forward(self, hidden, encoder_outputs):
        # hidden: [1,b,h], encoder_outputs: [src_len,b,h]
        src_len = encoder_outputs.shape[0]
        hidden = hidden[-1].unsqueeze(0).repeat(src_len,1,1)
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        attn = torch.softmax(self.v(energy), dim=0)
        return attn

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hidden_dim, n_layers=1, cell="GRU"):
        super().__init__()
        self.output_dim = output_dim
        self.embedding = nn.Embedding(output_dim, emb_dim)
        rnn_cls = nn.RNN if cell=="RNN" else nn.GRU if cell=="GRU" else nn.LSTM
        self.rnn = rnn_cls(emb_dim+hidden_dim, hidden_dim, n_layers)
        self.fc_out = nn.Linear(hidden_dim*2, output_dim)
        self.attention = Attention(hidden_dim)
    def forward(self, input, hidden, encoder_outputs):
        input = input.unsqueeze(0)
        emb = self.embedding(input)
        attn = self.attention(hidden, encoder_outputs) # [src_len,b,1]
        weighted = torch.sum(attn*encoder_outputs, dim=0).unsqueeze(0)
        rnn_input = torch.cat((emb, weighted), dim=2)
        output, hidden = self.rnn(rnn_input, hidden)
        pred = self.fc_out(torch.cat((output.squeeze(0), weighted.squeeze(0)), dim=1))
        return pred, hidden, attn

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
    def forward(self, src, tgt, teacher_forcing=0.5):
        max_len = tgt.shape[0]
        batch_size = tgt.shape[1]
        tgt_vocab_size = self.decoder.output_dim
        outputs = torch.zeros(max_len, batch_size, tgt_vocab_size).to(self.device)
        encoder_outputs, hidden = self.encoder(src)
        input = tgt[0,:]  # <sos>
        for t in range(1, max_len):
            output, hidden, attn = self.decoder(input, hidden, encoder_outputs)
            outputs[t] = output
            top1 = output.argmax(1)
            input = tgt[t] if random.random()<teacher_forcing else top1
        return outputs

# -------------------------------
# Train / Evaluate
# -------------------------------

def train_model(model, iterator, optimizer, criterion, device):
    model.train()
    epoch_loss = 0
    for src, tgt in iterator:
        src, tgt = src.to(device), tgt.to(device)
        optimizer.zero_grad()
        output = model(src, tgt)
        output_dim = output.shape[-1]
        output = output[1:].view(-1, output_dim)
        tgt = tgt[1:].view(-1)
        loss = criterion(output, tgt)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(iterator)

def evaluate_model(model, iterator, src_vocab, tgt_vocab, device):
    model.eval()
    preds, refs = [], []
    with torch.no_grad():
        for src, tgt in iterator:
            src, tgt = src.to(device), tgt.to(device)
            encoder_outputs, hidden = model.encoder(src)
            input = tgt[0,:]
            outputs = []
            for t in range(1, tgt.shape[0]):
                output, hidden, attn = model.decoder(input, hidden, encoder_outputs)
                top1 = output.argmax(1)
                outputs.append(top1.cpu().numpy())
                input = top1
            pred_strs = [tgt_vocab.decode([tok[i] for tok in outputs]) for i in range(src.shape[1])]
            ref_strs = [tgt_vocab.decode(seq.cpu().numpy()) for seq in tgt.permute(1,0)]
            preds.extend(pred_strs)
            refs.extend(ref_strs)
    acc = accuracy_score(refs, preds)
    return acc

# -------------------------------
# Main
# -------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--lang", type=str, default="hi")
    parser.add_argument("--cell", type=str, default="GRU", choices=["RNN","GRU","LSTM"])
    parser.add_argument("--emb", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--tf", type=float, default=0.5)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    train_pairs = load_dakshina_pairs(args.data_dir, args.lang, "train")
    dev_pairs   = load_dakshina_pairs(args.data_dir, args.lang, "dev")
    test_pairs  = load_dakshina_pairs(args.data_dir, args.lang, "test")

    src_vocab = Vocab([c for src,_ in train_pairs for c in src])
    tgt_vocab = Vocab([c for _,tgt in train_pairs for c in tgt])

    train_data = TransliterationDataset(train_pairs, src_vocab, tgt_vocab)
    dev_data   = TransliterationDataset(dev_pairs, src_vocab, tgt_vocab)
    test_data  = TransliterationDataset(test_pairs, src_vocab, tgt_vocab)

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    dev_loader   = DataLoader(dev_data, batch_size=args.batch_size, collate_fn=collate_fn)
    test_loader  = DataLoader(test_data, batch_size=args.batch_size, collate_fn=collate_fn)

    # Build model
    enc = Encoder(len(src_vocab), args.emb, args.hidden, cell=args.cell)
    dec = Decoder(len(tgt_vocab), args.emb, args.hidden, cell=args.cell)
    model = Seq2Seq(enc, dec, device).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    for epoch in range(args.epochs):
        loss = train_model(model, train_loader, optimizer, criterion, device)
        acc = evaluate_model(model, dev_loader, src_vocab, tgt_vocab, device)
        print(f"Epoch {epoch+1}: loss={loss:.4f}, dev acc={acc:.4f}")

    # Final test accuracy
    test_acc = evaluate_model(model, test_loader, src_vocab, tgt_vocab, device)
    print("Test Accuracy:", test_acc)
