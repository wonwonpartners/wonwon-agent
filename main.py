from __future__ import annotations

import argparse

from company_research_graph import run_company_research


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the company research",
    )
    parser.add_argument(
        "query",
        nargs="*",
        help="Keywords",
    )
    args = parser.parse_args()

    if not args.query:
        parser.print_help()
        return

    user_query = " ".join(args.query)
    result = run_company_research(user_query)


if __name__ == "__main__":
    main()
