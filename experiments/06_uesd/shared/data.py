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


_TASKS = {
    "copy": generate_copy_batch,
    "reversal": generate_reversal_batch,
    "sort": generate_sort_batch,
}


def generate_batch(task: str, batch_size: int, seq_len: int, vocab_size: int):
    if task not in _TASKS:
        raise ValueError(f"Unknown task {task!r}, expected one of {list(_TASKS)}")
    return _TASKS[task](batch_size, seq_len, vocab_size)
