from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "04_criticality_pruning"


def main() -> dict:
    card = json.loads((OUT / "CLAIM_CARD.json").read_text(encoding="utf-8"))
    synth = json.loads((OUT / "label_free_baseline_synthesis.json").read_text(encoding="utf-8"))
    checks = []
    checks.append({"name": "overall_vs_synflow_mean_delta", "ok": abs(card["primary_result"]["overall_vs_synflow_mean_delta"] - synth["overall"]["vs_synflow"]["mean"]) < 1e-12})
    checks.append({"name": "severe_98_vs_synflow_mean_delta", "ok": abs(card["primary_result"]["severe_98_vs_synflow_mean_delta"] - synth["severe_sparsity_98"]["vs_synflow"]["mean"]) < 1e-12})
    checks.append({"name": "severe_98_synflow_all_wins", "ok": card["primary_result"]["severe_98_vs_synflow_wins"] == card["primary_result"]["severe_98_vs_synflow_n"] == 4})
    checks.append({"name": "source_artifacts_exist", "ok": all((ROOT / path).exists() for path in card["source_artifacts"])})
    result = {"all_ok": all(c["ok"] for c in checks), "checks": checks}
    (OUT / "claim_card_audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    result = main()
    print(json.dumps(result, indent=2))
    if not result["all_ok"]:
        raise SystemExit(1)
