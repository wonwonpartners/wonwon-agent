from __future__ import annotations

import argparse
import logging

from company_research_graph import run_company_research


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    parser = argparse.ArgumentParser(
        description="Run the company research",
    )
    parser.add_argument(
        "query",
        nargs="*",
        help="Keywords",
    )
    parser.add_argument(
        "--force-report",
        action="store_true",
        help="Generate the report even when eval_state.ready_for_report is false.",
    )
    args = parser.parse_args()

    if not args.query:
        parser.print_help()
        return

    user_query = " ".join(args.query)
    result = run_company_research(
        user_query,
        force_report_generation=args.force_report,
    )
    print(result)


if __name__ == "__main__":
    main()
