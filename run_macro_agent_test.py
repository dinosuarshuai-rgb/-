from __future__ import annotations

import json
from datetime import datetime, timezone

from agents.macro_agent import MacroAgent
from models import MacroFactors


def main() -> None:
    agent = MacroAgent()
    query = "Iran geopolitical news latest escalation sanctions oil shipping"
    macro = MacroFactors(
        oil_price=1.2,
        dxy=0.2,
        etf_flow=0.1,
    )

    geo_signal, decision = agent.run(query=query, macro=macro, now=datetime.now(timezone.utc))

    print("=== GeoSignal ===")
    print(json.dumps(geo_signal.model_dump(mode="json"), indent=2, ensure_ascii=False))
    print("\n=== FinalDecision ===")
    print(json.dumps(decision.model_dump(mode="json"), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
