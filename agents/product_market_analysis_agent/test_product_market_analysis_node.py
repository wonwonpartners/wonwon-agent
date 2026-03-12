from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from product_market_analysis import (
    product_market_analysis_agent,
)


async def _run(startup_name: str) -> None:
    load_dotenv()
    result = await product_market_analysis_agent({"startup_name": startup_name})
    print(json.dumps(result, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the product market analysis node directly.",
    )
    parser.add_argument("startup_name", help="Startup name to analyze")
    args = parser.parse_args()

    asyncio.run(_run(args.startup_name))


if __name__ == "__main__":
    main()
