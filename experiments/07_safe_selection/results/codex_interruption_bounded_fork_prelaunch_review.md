CLEAN

Verified:

- Checkout: `21462bfaf0910e8f2a60d39d4e6c04a2ebc83590`
- Clean worktree
- Independently computed runner SHA-256: `0e72834961b2fe9944ff265ca4e7cedc954dbda202b5a2823ed762f953e19ea4`

1. **Round-3 blocker: FIXED.**
   The live fork routes existing durable state through `_restart_interruption_bounded_fork` before CUDA work ([runner:6373](<C:/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/07_safe_selection/exp_f1_bon_safe_selection.py:6373>)). The coordinator validates `started.json`, initializes/promotes the trusted head, and only then reconciles the ledger ([runner:5896](<C:/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/07_safe_selection/exp_f1_bon_safe_selection.py:5896>)). Low-level reconciliation also performs idempotent head initialization when a durable start exists ([runner:5689](<C:/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/07_safe_selection/exp_f1_bon_safe_selection.py:5689>)).

2. **Integration coverage is genuine.**
   Child processes invoke the actual runner script through its CLI worker branch and exit at the registered crash windows ([runner:7872](<C:/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/07_safe_selection/exp_f1_bon_safe_selection.py:7872>), [runner:8500](<C:/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/07_safe_selection/exp_f1_bon_safe_selection.py:8500>)). Recovery then calls the exact production coordinator used by the live fork, including repeated-restart idempotence and the event-present negative case ([runner:7981](<C:/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/07_safe_selection/exp_f1_bon_safe_selection.py:7981>), [runner:8034](<C:/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/07_safe_selection/exp_f1_bon_safe_selection.py:8034>)). It is not a duplicated recovery shim, though it intentionally avoids the full GPU launch path.

3. **Attestation-first ordering is preserved.**
   The CLI launch guard validates the registered runner and review hashes before calling the fork ([runner:8424](<C:/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/07_safe_selection/exp_f1_bon_safe_selection.py:8424>), [runner:8544](<C:/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/07_safe_selection/exp_f1_bon_safe_selection.py:8544>)). The fork then derives current manifest, generator, and terminal-probe identity before invoking restart recovery ([runner:6343](<C:/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/07_safe_selection/exp_f1_bon_safe_selection.py:6343>)).

4. **Final recovery probe: no new blocker.**
   Record-zero promotion requires no ledger events and rejects any history other than the exact canonical null record ([runner:5411](<C:/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/07_safe_selection/exp_f1_bon_safe_selection.py:5411>)). Therefore deleting an advanced head while events or advanced history remain closes rather than promoting. Kills before, during, or after atomic null-head promotion are idempotently recoverable. Ledger reconciliation precedes orphan reconciliation, so dangling-pair closure cannot be bypassed, while valid orphan promotion remains bound to the validated start identity.

Both preregistration slots remain `UNFILLED` ([PREREGISTRATION.md:2074](<C:/Users/devan/OneDrive/Desktop/Projects/neuro-ai-lab/experiments/07_safe_selection/PREREGISTRATION.md:2074>)), and the new hash remains launch-blocked until they are filled.

**This CLEAN verdict authorizes filling both hash slots.**
