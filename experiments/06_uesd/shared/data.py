"""Synthetic data generators for UESD experiments."""
import torch


def generate_copy_batch(batch_size: int, seq_len: int, vocab_size: int):
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    return input_ids, input_ids.clone()


def generate_reversal_batch(batch_size: int, seq_len: int, vocab_size: int):
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    target_ids = input_ids.flip(1)
    return input_ids, target_ids


def generate_sort_batch(batch_size: int, seq_len: int, vocab_size: int):
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    target_ids, _ = input_ids.sort(dim=1)
    return input_ids, target_ids


def generate_addition_batch(batch_size: int, seq_len: int, vocab_size: int):
    """Multi-digit addition: input is [a0,b0, a1,b1, ...], output is sum digits.

    Input: interleaved pairs (a_i, b_i) for i = 0..L/2-1, most-significant first.
    Output: L/2 digits of (A + B) mod base^(L/2), most-significant first.
    Carry propagation goes right-to-left — not solvable by position-wise mapping.
    Output length = seq_len (padded with 0 if seq_len is odd).
    """
    half = seq_len // 2
    a = torch.randint(0, vocab_size, (batch_size, half))
    b = torch.randint(0, vocab_size, (batch_size, half))
    input_ids = torch.zeros(batch_size, seq_len, dtype=torch.long)
    input_ids[:, 0::2] = a[:, :seq_len // 2 + seq_len % 2]
    input_ids[:, 1::2] = b[:, :seq_len // 2]

    carry = torch.zeros(batch_size, dtype=torch.long)
    result = torch.zeros(batch_size, half, dtype=torch.long)
    for i in range(half - 1, -1, -1):
        s = a[:, i] + b[:, i] + carry
        result[:, i] = s % vocab_size
        carry = s // vocab_size

    target_ids = torch.zeros(batch_size, seq_len, dtype=torch.long)
    target_ids[:, :half] = result
    return input_ids, target_ids


def generate_subtraction_batch(batch_size: int, seq_len: int, vocab_size: int):
    """Multi-digit subtraction: (A - B) mod base^half.

    Same input format as addition: interleaved [a0,b0,a1,b1,...].
    Borrow propagation goes right-to-left, analogous to carry in addition.
    """
    half = seq_len // 2
    a = torch.randint(0, vocab_size, (batch_size, half))
    b = torch.randint(0, vocab_size, (batch_size, half))
    input_ids = torch.zeros(batch_size, seq_len, dtype=torch.long)
    input_ids[:, 0::2] = a[:, :seq_len // 2 + seq_len % 2]
    input_ids[:, 1::2] = b[:, :seq_len // 2]

    borrow = torch.zeros(batch_size, dtype=torch.long)
    result = torch.zeros(batch_size, half, dtype=torch.long)
    for i in range(half - 1, -1, -1):
        diff = a[:, i].long() - b[:, i].long() - borrow
        borrow = (diff < 0).long()
        result[:, i] = diff % vocab_size

    target_ids = torch.zeros(batch_size, seq_len, dtype=torch.long)
    target_ids[:, :half] = result
    return input_ids, target_ids


def generate_dedup_batch(batch_size: int, seq_len: int, vocab_size: int):
    """Deduplicate + sort: output unique sorted elements, padded with 0.

    Non-bijective mapping: multiple inputs can map to the same output.
    Requires counting/grouping operations, not just routing.
    """
    input_ids = torch.randint(1, vocab_size, (batch_size, seq_len))
    target_list = []
    for i in range(batch_size):
        unique_sorted = input_ids[i].unique(sorted=True)
        padded = torch.zeros(seq_len, dtype=torch.long)
        n = min(len(unique_sorted), seq_len)
        padded[:n] = unique_sorted[:n]
        target_list.append(padded)
    target_ids = torch.stack(target_list)
    return input_ids, target_ids


def generate_prefix_sum_batch(batch_size: int, seq_len: int, vocab_size: int):
    """Prefix sum mod V: output[i] = sum(input[0:i+1]) mod vocab_size.

    Sequential depth is O(seq_len) — each output depends on ALL previous inputs.
    Different from addition: no interleaved pair format, no right-to-left carry.
    Instead, left-to-right cumulative accumulation.
    Loss should be computed over ALL seq_len positions (not just first half).
    """
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    prefix_sums = torch.zeros(batch_size, seq_len, dtype=torch.long)
    running = torch.zeros(batch_size, dtype=torch.long)
    for i in range(seq_len):
        running = (running + input_ids[:, i]) % vocab_size
        prefix_sums[:, i] = running
    return input_ids, prefix_sums


_TASKS = {
    "copy": generate_copy_batch,
    "reversal": generate_reversal_batch,
    "sort": generate_sort_batch,
    "addition": generate_addition_batch,
    "subtraction": generate_subtraction_batch,
    "dedup": generate_dedup_batch,
    "prefix_sum": generate_prefix_sum_batch,
}


def generate_batch(task: str, batch_size: int, seq_len: int, vocab_size: int):
    if task not in _TASKS:
        raise ValueError(f"Unknown task {task!r}, expected one of {list(_TASKS)}")
    return _TASKS[task](batch_size, seq_len, vocab_size)
