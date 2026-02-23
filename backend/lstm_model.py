import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa
from huggingface_hub import hf_hub_download

# ─── Model arhitektura ────────────────────────────────────────────────────────

class LSTMEmotionModel(nn.Module):
    def __init__(self, feat_dim=120, hidden=128, num_layers=2,
                 num_classes=8, dropout=0.3):
        super().__init__()
        self.hidden = hidden

        # 2-layer BiLSTM  →  l0: in=120, l1: in=256 (2×128)
        self.rnn = nn.LSTM(
            input_size=feat_dim,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Attention: Linear(256,128) → Tanh → Linear(128,1)
        self.attn = nn.Sequential(
            nn.Linear(hidden * 2, hidden),   # [128, 256]
            nn.Tanh(),
            nn.Linear(hidden, 1),            # [1, 128]
        )

        # FC: attn_out(256) + last_hidden(256) = 512 ulaz
        # fc.0=BN(512), fc.1=ReLU, fc.2=Linear(512,256),
        # fc.3=ReLU,    fc.4=Dropout, fc.5=Linear(256,8)
        self.fc = nn.Sequential(
            nn.LayerNorm(hidden * 4),
            nn.ReLU(),
            nn.Linear(hidden * 4, hidden * 2),  # fc.2 → shape [256, 512]
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden * 2, num_classes), # fc.5 → shape [8, 256]
        )

    def forward(self, x):
        # x: (B, T, 120)
        out, (h_n, _) = self.rnn(x)          # out: (B, T, 256)

        # Attention weighted sum
        scores  = self.attn(out).squeeze(-1)  # (B, T)
        weights = F.softmax(scores, dim=1).unsqueeze(-1)  # (B, T, 1)
        attn_out = (out * weights).sum(dim=1)             # (B, 256)

        # Zadnji hidden state zadnjeg layera (fwd + bwd)
        h_fwd = h_n[-2]                                   # (B, 128)
        h_bwd = h_n[-1]                                   # (B, 128)
        last_h = torch.cat([h_fwd, h_bwd], dim=-1)       # (B, 256)

        combined = torch.cat([attn_out, last_h], dim=-1)  # (B, 512)
        return self.fc(combined)


# ─── Singleton cache ──────────────────────────────────────────────────────────

_lstm_model = None

def _get_lstm_model():
    global _lstm_model
    if _lstm_model is None:
        path = hf_hub_download(
            repo_id="svantip123/LstmAudioEmotion",
            filename="lstm_best.pt"
        )
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        model = LSTMEmotionModel()
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        _lstm_model = model
    return _lstm_model


# ─── Predikcija ───────────────────────────────────────────────────────────────

ID2LABEL = {
    0: "angry", 1: "calm",    2: "disgust",  3: "fearful",
    4: "happy", 5: "neutral", 6: "sad",      7: "surprised",
}

# Config iz checkpointa
_CFG = dict(sr=16000, n_mfcc=40, n_fft=400,
            hop=160, win=400, fixed_t=461, feat_dim=120)


def predict_lstm_audio(audio_path: str) -> dict:
    # 1) Učitaj audio
    wav, _ = librosa.load(audio_path, sr=_CFG["sr"])

    # 2) MFCC + Δ + ΔΔ  →  (T, 120)
    mfcc    = librosa.feature.mfcc(
        y=wav, sr=_CFG["sr"], n_mfcc=_CFG["n_mfcc"],
        n_fft=_CFG["n_fft"], hop_length=_CFG["hop"],
        win_length=_CFG["win"]
    )
    delta   = librosa.feature.delta(mfcc)
    delta2  = librosa.feature.delta(mfcc, order=2)
    feats   = np.concatenate([mfcc, delta, delta2], axis=0).T  # (T, 120)

    # 3) Pad / truncate na fixed_t=461
    T = feats.shape[0]
    if T < _CFG["fixed_t"]:
        feats = np.concatenate(
            [feats, np.zeros((_CFG["fixed_t"] - T, _CFG["feat_dim"]))], axis=0
        )
    else:
        feats = feats[: _CFG["fixed_t"]]

    # 4) Per-utterance normalizacija
    feats = (feats - feats.mean(0, keepdims=True)) / (feats.std(0, keepdims=True) + 1e-8)

    # 5) Tensor  (1, 461, 120)
    x = torch.tensor(feats, dtype=torch.float32).unsqueeze(0)

    # 6) Inferencija
    model = _get_lstm_model()
    with torch.no_grad():
        probs = F.softmax(model(x), dim=-1).squeeze(0).tolist()

    pred = int(np.argmax(probs))
    return {
        "model":         "LstmAudioEmotion",
        "emotion":       ID2LABEL[pred],
        "confidence":    round(probs[pred], 4),
        "probabilities": {ID2LABEL[i]: round(p, 4) for i, p in enumerate(probs)},
    }
