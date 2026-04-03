import sys
from app import translate_en_hi, DEVICE
import torch
import json
import os
from app import Vocabulary, build_model, beam_search

wp = "saved_models/en_hi_seq2seq.pt"
vp = "saved_models/en_hi_vocab.json"
vd = json.load(open(vp))
tv = Vocabulary()
tv.w2i = vd["tw2i"]; tv.i2w = {int(k):v for k,v in vd["ti2w"].items()}; tv.n = vd["tn"]
print("i2w snippet:", {k: tv.i2w[k] for k in list(tv.i2w)[:10]})
