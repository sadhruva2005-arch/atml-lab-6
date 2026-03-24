"""
app.py — Encoder-Decoder NLP Suite
All 4 tasks in one file: En→Hi (LSTM), En→Es (MarianMT), Summarization (BART)

Install: pip install flask flask-cors torch transformers datasets sacrebleu sentencepiece
Run:     python app.py
"""

import os, json, random
import torch
import torch.nn as nn
from flask import Flask, request, jsonify
from flask_cors import CORS

app   = Flask(__name__)
CORS(app)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────────────────────
#  TASK 1 & 2 — Custom LSTM Seq2Seq (En → Hi)
# ─────────────────────────────────────────────

class Vocabulary:
    PAD, SOS, EOS, UNK = 0, 1, 2, 3
    def __init__(self):
        self.w2i = {"<PAD>":0,"<SOS>":1,"<EOS>":2,"<UNK>":3}
        self.i2w = {v:k for k,v in self.w2i.items()}
        self.n   = 4
    def add(self, sentence):
        for w in sentence.split():
            if w not in self.w2i:
                self.w2i[w] = self.n
                self.i2w[self.n] = w
                self.n += 1
    def to_tensor(self, sentence, max_len=50):
        ids = [self.w2i.get(w, self.UNK) for w in sentence.split()][:max_len] + [self.EOS]
        return torch.tensor(ids, dtype=torch.long, device=DEVICE)

class Encoder(nn.Module):
    def __init__(self, vsz, emb, hid, layers, drop):
        super().__init__()
        self.emb  = nn.Embedding(vsz, emb, padding_idx=0)
        self.lstm = nn.LSTM(emb, hid, layers, batch_first=True,
                            dropout=drop if layers > 1 else 0)
        self.drop = nn.Dropout(drop)
    def forward(self, x):
        return self.lstm(self.drop(self.emb(x)))

class Decoder(nn.Module):
    def __init__(self, vsz, emb, hid, layers, drop):
        super().__init__()
        self.emb  = nn.Embedding(vsz, emb, padding_idx=0)
        self.lstm = nn.LSTM(emb+hid, hid, layers, batch_first=True,
                            dropout=drop if layers > 1 else 0)
        self.fc   = nn.Linear(hid*2, vsz)
        self.drop = nn.Dropout(drop)
    def forward(self, tok, h, c, enc_out):
        e   = self.drop(self.emb(tok.unsqueeze(1)))
        ctx = enc_out.mean(1, keepdim=True)
        o, (h,c) = self.lstm(torch.cat([e,ctx],2),(h,c))
        return self.fc(torch.cat([o,ctx],2)).squeeze(1), h, c

class Seq2Seq(nn.Module):
    def __init__(self, enc, dec):
        super().__init__()
        self.enc, self.dec = enc, dec
    def forward(self, src, trg, tf=0.5):
        B,T   = trg.shape
        outs  = torch.zeros(B,T,self.dec.fc.out_features,device=DEVICE)
        eo,(h,c) = self.enc(src)
        di    = trg[:,0]
        for t in range(1,T):
            pred,h,c = self.dec(di,h,c,eo)
            outs[:,t] = pred
            di = trg[:,t] if random.random()<tf else pred.argmax(1)
        return outs

def build_lstm_model(sv, tv, emb=256, hid=512, layers=2, drop=0.3):
    enc = Encoder(sv, emb, hid, layers, drop)
    dec = Decoder(tv, emb, hid, layers, drop)
    return Seq2Seq(enc, dec).to(DEVICE)

def translate_en_hi(text):
    wp = "saved_models/en_hi_seq2seq.pt"
    vp = "saved_models/en_hi_vocab.json"
    if not (os.path.exists(wp) and os.path.exists(vp)):
        return "⚠️ Model not trained yet — run: python app.py --train-hi"

    vd = json.load(open(vp))
    sv = Vocabulary(); sv.w2i=vd["sw2i"]; sv.i2w={int(k):v for k,v in vd["si2w"].items()}; sv.n=vd["sn"]
    tv = Vocabulary(); tv.w2i=vd["tw2i"]; tv.i2w={int(k):v for k,v in vd["ti2w"].items()}; tv.n=vd["tn"]

    model = build_lstm_model(sv.n, tv.n)
    model.load_state_dict(torch.load(wp, map_location=DEVICE))
    model.eval()

    src = sv.to_tensor(text.lower()).unsqueeze(0)
    with torch.no_grad():
        eo,(h,c) = model.enc(src)
        di = torch.tensor([tv.SOS], device=DEVICE)
        words = []
        for _ in range(60):
            pred,h,c = model.dec(di,h,c,eo)
            t = pred.argmax(1)
            if t.item()==tv.EOS: break
            words.append(tv.i2w.get(t.item(),""))
            di = t
    return " ".join(words) or "No output — try more training epochs"


# ─────────────────────────────────────────────
#  TASK 1 & 2 — Training helper (run once)
# ─────────────────────────────────────────────

def train_en_hi(data_path="data/Hindi_English_Truncated_Corpus.csv", epochs=20, batch=64):
    import csv
    from torch.utils.data import Dataset, DataLoader

    print("Loading data …")
    pairs = []
    with open(data_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            en = row.get("english_sentence","").strip().lower()
            hi = row.get("hindi_sentence","").strip()
            if en and hi: pairs.append((en, hi))
    pairs = pairs[:5000]
    random.shuffle(pairs)
    split = int(0.9*len(pairs))
    train_pairs, test_pairs = pairs[:split], pairs[split:]

    sv, tv = Vocabulary(), Vocabulary()
    for en,hi in train_pairs: sv.add(en); tv.add(hi)
    print(f"Vocab: EN={sv.n} HI={tv.n} | Train={len(train_pairs)} Test={len(test_pairs)}")

    class DS(Dataset):
        def __getitem__(self, i):
            en,hi = train_pairs[i]
            return sv.to_tensor(en), tv.to_tensor(hi)
        def __len__(self): return len(train_pairs)

    def collate(b):
        s,t = zip(*b)
        return (nn.utils.rnn.pad_sequence(s,True),
                nn.utils.rnn.pad_sequence(t,True))

    dl    = DataLoader(DS(), batch_size=batch, shuffle=True, collate_fn=collate)
    model = build_lstm_model(sv.n, tv.n)
    opt   = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit  = nn.CrossEntropyLoss(ignore_index=0)
    best  = float("inf")

    for ep in range(1, epochs+1):
        model.train(); loss_sum = 0
        for src,trg in dl:
            src,trg = src.to(DEVICE),trg.to(DEVICE)
            opt.zero_grad()
            out = model(src,trg)
            loss = crit(out[:,1:].reshape(-1,tv.n), trg[:,1:].reshape(-1))
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1); opt.step()
            loss_sum += loss.item()
        avg = loss_sum/len(dl)
        print(f"Epoch {ep}/{epochs}  loss={avg:.4f}")
        if avg < best:
            best = avg
            os.makedirs("saved_models", exist_ok=True)
            torch.save(model.state_dict(), "saved_models/en_hi_seq2seq.pt")
            json.dump({"sw2i":sv.w2i,"si2w":sv.i2w,"sn":sv.n,
                       "tw2i":tv.w2i,"ti2w":tv.i2w,"tn":tv.n},
                      open("saved_models/en_hi_vocab.json","w"), ensure_ascii=False)
    print("Training done. Model saved.")


# ─────────────────────────────────────────────
#  TASK 3 — MarianMT (En → Es)
# ─────────────────────────────────────────────

_es_pipe = None
def translate_en_es(text):
    global _es_pipe
    if _es_pipe is None:
        from transformers import pipeline
        print("Loading MarianMT …")
        _es_pipe = pipeline("translation_en_to_es", model="Helsinki-NLP/opus-mt-en-es")
    return _es_pipe(text, max_length=512)[0]["translation_text"]


# ─────────────────────────────────────────────
#  TASK 4 — BART Summarization
# ─────────────────────────────────────────────

_bart_pipe = None
def summarize(text, max_len=150, min_len=40):
    global _bart_pipe
    if _bart_pipe is None:
        from transformers import pipeline
        print("Loading BART …")
        _bart_pipe = pipeline("summarization", model="facebook/bart-large-cnn")
    result = _bart_pipe(text, max_length=max_len, min_length=min_len,
                        do_sample=False)[0]["summary_text"]
    wc_in  = len(text.split())
    wc_out = len(result.split())
    return result, wc_in, wc_out


# ─────────────────────────────────────────────
#  Flask Routes
# ─────────────────────────────────────────────

@app.get("/api/health")
def health():
    return jsonify({"status": "ok"})

@app.post("/api/translate/en-hi")
def route_en_hi():
    text = (request.json or {}).get("text","").strip()
    if not text: return jsonify({"error": "Provide 'text'"}), 400
    return jsonify({"translation": translate_en_hi(text)})

@app.post("/api/translate/en-es")
def route_en_es():
    text = (request.json or {}).get("text","").strip()
    if not text: return jsonify({"error": "Provide 'text'"}), 400
    return jsonify({"translation": translate_en_es(text)})

@app.post("/api/summarize")
def route_summarize():
    body    = request.json or {}
    text    = body.get("text","").strip()
    max_len = int(body.get("max_length", 150))
    min_len = int(body.get("min_length", 40))
    if not text: return jsonify({"error": "Provide 'text'"}), 400
    if len(text.split()) < 30: return jsonify({"error": "Need at least 30 words"}), 400
    summary, wc_in, wc_out = summarize(text, max_len, min_len)
    return jsonify({"summary": summary, "original_words": wc_in, "summary_words": wc_out,
                    "compression": f"{round((1-wc_out/wc_in)*100,1)}%"})


# ─────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    if "--train-hi" in sys.argv:
        # python app.py --train-hi
        train_en_hi()
    else:
        app.run(host="0.0.0.0", port=8080, debug=True)
