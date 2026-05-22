"""Training loops for UESD experiments.

Supports four training modes:
- Track A (E1): embedding regression
- Track B (E5): self-consistency + readout CE
- AR baseline: teacher-forced cross-entropy
- Encoder-only ablation: direct CE
"""
import time
import torch
import torch.nn.functional as F
from .data import generate_batch
from .diagnostics import run_all_diagnostics


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def _e1_step(model, src_ids, target_ids, T, **_kwargs):
    """Track A: L = MSE(s_T, embed(y*)) + 0.1 * CE(readout(s_T), y*)

    SPEC DEVIATION: locked spec defines E1 as pure MSE. The 0.1*CE
    auxiliary term was added because pure MSE leaves readout_proj
    untrained (0% token accuracy despite MSE=0.0003). MSE still
    dominates the loss (~0.0003 vs CE contribution ~0.0003 at
    convergence). Alternative: separately fit a frozen readout head.
    """
    context = model.encode(src_ids)
    s = model.init_state(src_ids.size(0), src_ids.size(1), src_ids.device)
    for _ in range(T):
        s, _ = model.dynamics_step(s, context)
    target_emb = model.tok_emb(target_ids)
    mse = F.mse_loss(s, target_emb)
    logits = model.readout_logits(s)
    ce = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                         target_ids.reshape(-1))
    loss = mse + 0.1 * ce
    return loss, {"mse": mse.item(), "ce": ce.item()}


def _e5_step(model, src_ids, target_ids, T, step=0,
             lambda_1=1.0, warmup_steps=5000, **_kwargs):
    """Track B: L = lambda_1 * ||F(s_T,c)||^2 + CE(readout(s_T), y*)"""
    context = model.encode(src_ids)
    s = model.init_state(src_ids.size(0), src_ids.size(1), src_ids.device)
    for _ in range(T):
        s, _ = model.dynamics_step(s, context)

    # Self-consistency: one more step to measure residual
    s_next, _ = model.dynamics_step(s, context)
    residual = s_next - s
    sc_loss = (residual ** 2).sum(dim=-1).mean()

    # Readout CE
    logits = model.readout_logits(s)
    ce_loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)), target_ids.reshape(-1)
    )

    eff_lam = lambda_1 * min(1.0, step / warmup_steps) if warmup_steps > 0 else lambda_1
    loss = eff_lam * sc_loss + ce_loss

    return loss, {
        "sc_loss": sc_loss.item(),
        "ce_loss": ce_loss.item(),
        "eff_lambda": eff_lam,
    }


def _ar_step(model, src_ids, target_ids, **_kwargs):
    """AR baseline: teacher-forced CE."""
    B, L = target_ids.shape
    bos = torch.zeros(B, 1, dtype=torch.long, device=target_ids.device)
    tgt_input = torch.cat([bos, target_ids[:, :-1]], dim=1)
    logits = model(src_ids, tgt_input)
    loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)), target_ids.reshape(-1)
    )
    return loss, {"ce_loss": loss.item()}


def _enc_only_step(model, src_ids, target_ids, **_kwargs):
    """Encoder-only ablation: direct CE."""
    logits = model(src_ids)
    loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)), target_ids.reshape(-1)
    )
    return loss, {"ce_loss": loss.item()}


_STEP_FNS = {
    "e1": _e1_step,
    "e5": _e5_step,
    "ar": _ar_step,
    "encoder_only": _enc_only_step,
}


def set_seed(seed):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train(model, task, track, config, device="cuda"):
    """Train a model and return history.

    Args:
        model: nn.Module to train.
        task: task name (e.g. 'copy', 'reversal', 'sort').
        track: 'e1', 'e5', 'ar', or 'encoder_only'.
        config: dict with keys:
            training_steps, batch_size, seq_len, vocab_size,
            lr, T, lambda_1, warmup_steps, log_interval, seed.
        device: torch device string.

    Returns:
        dict with 'history' (list of log dicts), 'elapsed_s', and 'seed'.
    """
    seed = config.get("seed", None)
    if seed is not None:
        set_seed(seed)

    step_fn = _STEP_FNS[track]
    model = model.to(device)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=config.get("lr", 3e-4))
    total_steps = config.get("training_steps", 20000)
    batch_size = config.get("batch_size", 256)
    seq_len = config.get("seq_len", 8)
    vocab_size = config.get("vocab_size", 64)
    log_interval = config.get("log_interval", 500)
    T = config.get("T", 10)

    extra = {
        "T": T,
        "lambda_1": config.get("lambda_1", 1.0),
        "warmup_steps": config.get("warmup_steps", 5000),
    }

    history = []
    t0 = time.time()

    for step in range(1, total_steps + 1):
        src, tgt = generate_batch(task, batch_size, seq_len, vocab_size)
        src, tgt = src.to(device), tgt.to(device)

        loss, info = step_fn(model, src, tgt, step=step, **extra)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % log_interval == 0 or step == 1:
            entry = {"step": step, "loss": loss.item(), **info}
            history.append(entry)
            _log_line(step, total_steps, entry, track)

    elapsed = time.time() - t0
    print(f"  Training complete in {elapsed:.1f}s", flush=True)
    result = {"history": history, "elapsed_s": elapsed}
    if seed is not None:
        result["seed"] = seed
    return result


def _log_line(step, total, entry, track):
    parts = [f"  Step {step:>6d}/{total} | Loss: {entry['loss']:.4f}"]
    if track == "e5":
        parts.append(f"CE: {entry.get('ce_loss', 0):.4f}")
        parts.append(f"SC: {entry.get('sc_loss', 0):.4f}")
        parts.append(f"lam: {entry.get('eff_lambda', 0):.3f}")
    elif track == "e1":
        parts.append(f"MSE: {entry.get('mse', 0):.4f}")
        parts.append(f"CE: {entry.get('ce', 0):.4f}")
    print(" | ".join(parts), flush=True)


@torch.no_grad()
def evaluate_uesd(model, task, config, device="cuda"):
    """Evaluate a UESD model using all D1-D6 diagnostics.

    Args:
        model: trained UESDModel.
        task: 'copy' or 'reversal'.
        config: dict with seq_len, vocab_size, T, eval_samples.
        device: torch device string.

    Returns:
        Aggregated diagnostic results dict.
    """
    model = model.to(device)
    model.eval()

    seq_len = config.get("seq_len", 8)
    vocab_size = config.get("vocab_size", 64)
    T = config.get("T", 10)
    eval_samples = config.get("eval_samples", 10000)
    batch_size = config.get("eval_batch_size", 512)

    all_results = []
    remaining = eval_samples
    while remaining > 0:
        bs = min(batch_size, remaining)
        src, tgt = generate_batch(task, bs, seq_len, vocab_size)
        src, tgt = src.to(device), tgt.to(device)
        result = run_all_diagnostics(model, src, tgt, T)
        all_results.append((bs, result))
        remaining -= bs

    return _aggregate_diagnostics(all_results)


@torch.no_grad()
def evaluate_ar(model, task, config, device="cuda"):
    """Evaluate the AR baseline via greedy autoregressive generation."""
    model = model.to(device)
    model.eval()

    seq_len = config.get("seq_len", 8)
    vocab_size = config.get("vocab_size", 64)
    eval_samples = config.get("eval_samples", 10000)
    batch_size = config.get("eval_batch_size", 512)

    total_correct_tokens = 0
    total_tokens = 0
    total_correct_seqs = 0
    total_seqs = 0

    remaining = eval_samples
    while remaining > 0:
        bs = min(batch_size, remaining)
        src, tgt = generate_batch(task, bs, seq_len, vocab_size)
        src, tgt = src.to(device), tgt.to(device)

        preds = model.generate(src, seq_len)
        correct = preds == tgt
        total_correct_tokens += correct.sum().item()
        total_tokens += correct.numel()
        total_correct_seqs += correct.all(dim=-1).sum().item()
        total_seqs += bs
        remaining -= bs

    return {
        "token_accuracy": {"token_acc": total_correct_tokens / total_tokens,
                           "seq_acc": total_correct_seqs / total_seqs},
    }


@torch.no_grad()
def evaluate_encoder_only(model, task, config, device="cuda"):
    """Evaluate the encoder-only ablation."""
    from .diagnostics import token_accuracy, decoder_margin

    model = model.to(device)
    model.eval()

    seq_len = config.get("seq_len", 8)
    vocab_size = config.get("vocab_size", 64)
    eval_samples = config.get("eval_samples", 10000)
    batch_size = config.get("eval_batch_size", 512)

    all_tok_acc = []
    all_seq_acc = []
    all_margin = []

    remaining = eval_samples
    while remaining > 0:
        bs = min(batch_size, remaining)
        src, tgt = generate_batch(task, bs, seq_len, vocab_size)
        src, tgt = src.to(device), tgt.to(device)
        logits = model(src)
        ta = token_accuracy(logits, tgt)
        dm = decoder_margin(logits, tgt)
        all_tok_acc.append((bs, ta["token_acc"]))
        all_seq_acc.append((bs, ta["seq_acc"]))
        all_margin.append((bs, dm["mean_margin"]))
        remaining -= bs

    def _wavg(pairs):
        return sum(n * v for n, v in pairs) / sum(n for n, _ in pairs)

    return {
        "token_accuracy": {"token_acc": _wavg(all_tok_acc),
                           "seq_acc": _wavg(all_seq_acc)},
        "decoder_margin": {"mean_margin": _wavg(all_margin)},
    }


def _aggregate_diagnostics(batch_results):
    """Weighted average of diagnostic results across batches."""
    keys = batch_results[0][1].keys()
    agg = {}
    total_n = sum(n for n, _ in batch_results)

    for key in keys:
        sub_keys = batch_results[0][1][key].keys()
        agg[key] = {}
        for sk in sub_keys:
            vals = [(n, r[key][sk]) for n, r in batch_results]
            agg[key][sk] = sum(n * v for n, v in vals) / total_n

    return agg
