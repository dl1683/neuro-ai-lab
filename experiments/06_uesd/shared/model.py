"""Model definitions for UESD experiments: core UESD, baselines, and ablations."""
import torch
import torch.nn as nn
import torch.nn.utils.parametrizations as parametrizations


def _make_encoder(vocab_size, d_model, n_heads, d_ff, n_enc_layers):
    layer = nn.TransformerEncoderLayer(
        d_model, n_heads, dim_feedforward=d_ff, batch_first=True, norm_first=True,
    )
    return nn.TransformerEncoder(layer, num_layers=n_enc_layers)


class UESDModel(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, d_ff, n_enc_layers, max_len):
        super().__init__()
        self.d_model = d_model
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_enc = nn.Embedding(max_len, d_model)
        self.pos_dec = nn.Embedding(max_len, d_model)
        self.encoder = _make_encoder(vocab_size, d_model, n_heads, d_ff, n_enc_layers)
        self.dynamics = nn.TransformerDecoderLayer(
            d_model, n_heads, dim_feedforward=d_ff, batch_first=True, norm_first=True,
        )
        # Spectral norm on FFN linears inside the dynamics block
        parametrizations.spectral_norm(self.dynamics.linear1)
        parametrizations.spectral_norm(self.dynamics.linear2)
        self.readout_proj = nn.Linear(d_model, d_model)
        self.tau = 0.1

    def encode(self, src_ids):
        B, L = src_ids.shape
        pos = torch.arange(L, device=src_ids.device).unsqueeze(0)
        x = self.tok_emb(src_ids) + self.pos_enc(pos)
        return self.encoder(x)

    def init_state(self, batch_size, seq_len, device):
        pos = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        return self.pos_dec(pos)

    def dynamics_step(self, s, context):
        s_new = self.dynamics(s, context)
        update_norm = (s_new - s).norm(dim=-1).mean()
        return s_new, update_norm

    def unroll(self, src_ids, T):
        context = self.encode(src_ids)
        B, L_out = src_ids.shape
        s = self.init_state(B, L_out, src_ids.device)
        norms = []
        for _ in range(T):
            s, norm = self.dynamics_step(s, context)
            norms.append(norm)
        return s, norms

    def readout_logits(self, s):
        h = self.readout_proj(s)
        W = self.tok_emb.weight
        h = torch.nn.functional.normalize(h, dim=-1)
        W = torch.nn.functional.normalize(W, dim=-1)
        return torch.matmul(h, W.t()) / self.tau

    def forward(self, src_ids, T):
        s, _ = self.unroll(src_ids, T)
        return self.readout_logits(s)


class UntiedUESDModel(nn.Module):
    """UESD ablation with separate (untied) decoder layers per dynamics step."""

    def __init__(self, vocab_size, d_model, n_heads, d_ff, n_enc_layers, max_len, T):
        super().__init__()
        self.d_model = d_model
        self.T = T
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_enc = nn.Embedding(max_len, d_model)
        self.pos_dec = nn.Embedding(max_len, d_model)
        self.encoder = _make_encoder(vocab_size, d_model, n_heads, d_ff, n_enc_layers)
        self.dynamics_layers = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model, n_heads, dim_feedforward=d_ff, batch_first=True, norm_first=True,
            )
            for _ in range(T)
        ])
        for layer in self.dynamics_layers:
            parametrizations.spectral_norm(layer.linear1)
            parametrizations.spectral_norm(layer.linear2)
        self.readout_proj = nn.Linear(d_model, d_model)
        self.tau = 0.1

    def encode(self, src_ids):
        B, L = src_ids.shape
        pos = torch.arange(L, device=src_ids.device).unsqueeze(0)
        x = self.tok_emb(src_ids) + self.pos_enc(pos)
        return self.encoder(x)

    def init_state(self, batch_size, seq_len, device):
        pos = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        return self.pos_dec(pos)

    def unroll(self, src_ids):
        context = self.encode(src_ids)
        B, L_out = src_ids.shape
        s = self.init_state(B, L_out, src_ids.device)
        norms = []
        for layer in self.dynamics_layers:
            s_new = layer(s, context)
            norms.append((s_new - s).norm(dim=-1).mean())
            s = s_new
        return s, norms

    def readout_logits(self, s):
        h = self.readout_proj(s)
        W = self.tok_emb.weight
        h = torch.nn.functional.normalize(h, dim=-1)
        W = torch.nn.functional.normalize(W, dim=-1)
        return torch.matmul(h, W.t()) / self.tau

    def forward(self, src_ids):
        s, _ = self.unroll(src_ids)
        return self.readout_logits(s)


class ARBaseline(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, d_ff, n_enc_layers, n_dec_layers, max_len):
        super().__init__()
        self.d_model = d_model
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_enc = nn.Embedding(max_len, d_model)
        self.pos_dec = nn.Embedding(max_len, d_model)
        self.encoder = _make_encoder(vocab_size, d_model, n_heads, d_ff, n_enc_layers)
        dec_layer = nn.TransformerDecoderLayer(
            d_model, n_heads, dim_feedforward=d_ff, batch_first=True, norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=n_dec_layers)
        self.head = nn.Linear(d_model, vocab_size)

    def encode(self, src_ids):
        B, L = src_ids.shape
        pos = torch.arange(L, device=src_ids.device).unsqueeze(0)
        x = self.tok_emb(src_ids) + self.pos_enc(pos)
        return self.encoder(x)

    def forward(self, src_ids, tgt_ids):
        memory = self.encode(src_ids)
        B, L = tgt_ids.shape
        pos = torch.arange(L, device=tgt_ids.device).unsqueeze(0)
        tgt = self.tok_emb(tgt_ids) + self.pos_dec(pos)
        mask = nn.Transformer.generate_square_subsequent_mask(L, device=tgt_ids.device)
        out = self.decoder(tgt, memory, tgt_mask=mask)
        return self.head(out)

    @torch.no_grad()
    def generate(self, src_ids, max_len):
        memory = self.encode(src_ids)
        B = src_ids.size(0)
        device = src_ids.device
        tokens = torch.zeros(B, 1, dtype=torch.long, device=device)
        for _ in range(max_len):
            L = tokens.size(1)
            pos = torch.arange(L, device=device).unsqueeze(0)
            tgt = self.tok_emb(tokens) + self.pos_dec(pos)
            mask = nn.Transformer.generate_square_subsequent_mask(L, device=device)
            out = self.decoder(tgt, memory, tgt_mask=mask)
            logits = self.head(out[:, -1, :])
            next_tok = logits.argmax(dim=-1, keepdim=True)
            tokens = torch.cat([tokens, next_tok], dim=1)
        return tokens[:, 1:]


class EncoderOnlyAblation(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, d_ff, n_enc_layers, max_len):
        super().__init__()
        self.d_model = d_model
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_enc = nn.Embedding(max_len, d_model)
        self.encoder = _make_encoder(vocab_size, d_model, n_heads, d_ff, n_enc_layers)
        self.proj = nn.Linear(d_model, d_model)
        self.tau = 0.1

    def encode(self, src_ids):
        B, L = src_ids.shape
        pos = torch.arange(L, device=src_ids.device).unsqueeze(0)
        x = self.tok_emb(src_ids) + self.pos_enc(pos)
        return self.encoder(x)

    def readout_logits(self, h):
        h = self.proj(h)
        W = self.tok_emb.weight
        h = torch.nn.functional.normalize(h, dim=-1)
        W = torch.nn.functional.normalize(W, dim=-1)
        return torch.matmul(h, W.t()) / self.tau

    def forward(self, src_ids):
        h = self.encode(src_ids)
        return self.readout_logits(h)


def default_config():
    return {
        "vocab_size": 64,
        "d_model": 128,
        "n_heads": 4,
        "d_ff": 512,
        "n_enc_layers": 2,
        "n_dec_layers": 2,
        "max_len": 32,
        "T": 10,
    }


def build_models(config=None):
    c = default_config()
    if config:
        c.update(config)
    return {
        "uesd": UESDModel(
            c["vocab_size"], c["d_model"], c["n_heads"], c["d_ff"],
            c["n_enc_layers"], c["max_len"],
        ),
        "uesd_untied": UntiedUESDModel(
            c["vocab_size"], c["d_model"], c["n_heads"], c["d_ff"],
            c["n_enc_layers"], c["max_len"], c["T"],
        ),
        "ar_baseline": ARBaseline(
            c["vocab_size"], c["d_model"], c["n_heads"], c["d_ff"],
            c["n_enc_layers"], c["n_dec_layers"], c["max_len"],
        ),
        "encoder_only": EncoderOnlyAblation(
            c["vocab_size"], c["d_model"], c["n_heads"], c["d_ff"],
            c["n_enc_layers"], c["max_len"],
        ),
    }
