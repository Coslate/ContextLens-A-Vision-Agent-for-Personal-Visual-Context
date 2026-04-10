"""Interactive query CLI for the ContextLens memory store.

Queries the SQLite memory store populated by run_demo.py.

Usage:
    # Single query
    python -m scripts.run_query "all receipts"
    python -m scripts.run_query "conversations mentioning a meeting"
    python -m scripts.run_query "whiteboard photos from project Alpha"
    python -m scripts.run_query "images needing clarification"

    # Interactive mode (no argument)
    python -m scripts.run_query
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def run_single_query(query_text: str, db_path: str):
    """Run a single query and return QueryResult."""
    from contextlens.memory_store import MemoryStore
    from contextlens.query import query

    with MemoryStore(db_path=db_path) as store:
        return query(store, query_text)


def print_results(query_text: str, qr) -> None:
    """Pretty-print query results."""
    from contextlens.query import QueryResult

    results = qr.results if isinstance(qr, QueryResult) else qr
    matched = qr.matched if isinstance(qr, QueryResult) else True

    print(f'\nQ: "{query_text}"')

    if not matched:
        print(f"   \u26a0 No specific pattern matched \u2014 showing all {len(results)} images.")
        print(f"   Supported patterns: {QueryResult.SUPPORTED_PATTERNS}\n")
    else:
        print(f"   {len(results)} result(s)\n")

    if not results:
        print("   (no matching images)\n")
        return

    for r in results:
        image_id = r.get("image_id", "?")
        img_type = r.get("type", "?")
        summary = r.get("summary", "")
        group = r.get("group_id") or "-"
        conf = r.get("type_confidence")
        conf_str = f"{conf:.2f}" if conf is not None else "?"
        needs = r.get("needs_clarification", False)

        print(f"   [{image_id}]  type={img_type}  conf={conf_str}  group={group}")
        if summary:
            print(f"      summary: {summary}")
        if needs:
            print("      ⚠ needs clarification")
        print()


def interactive_mode(db_path: str) -> None:
    """Run interactive query loop."""
    print("=" * 60)
    print("ContextLens Query Interface")
    print("=" * 60)
    print(f"Database: {db_path}")
    print('Type a query and press Enter. Type "quit" to exit.\n')
    print("Example queries:")
    print('  - "all receipts"')
    print('  - "whiteboard photos from project Alpha"')
    print('  - "conversations mentioning a meeting"')
    print('  - "images needing clarification"')
    print('  - "all images from last week"')
    print()

    while True:
        try:
            query_text = input("query> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not query_text:
            continue
        if query_text.lower() in ("quit", "exit", "q"):
            print("Bye!")
            break

        results = run_single_query(query_text, db_path)
        print_results(query_text, results)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Query the ContextLens memory store",
    )
    parser.add_argument(
        "query",
        nargs="?",
        help="Query string. If omitted, enters interactive mode.",
    )
    parser.add_argument(
        "--db",
        default=str(PROJECT_ROOT / "contextlens_demo.db"),
        help="Path to SQLite database (default: contextlens_demo.db)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="as_json",
        help="Output results as JSON",
    )

    args = parser.parse_args()

    # Check DB exists
    if not Path(args.db).exists():
        print(f"Error: Database not found at {args.db}")
        print("Run 'python -m scripts.run_demo' first to populate the memory store.")
        sys.exit(1)

    if args.query:
        qr = run_single_query(args.query, args.db)
        if args.as_json:
            print(json.dumps(qr.results, indent=2, default=str))
        else:
            print_results(args.query, qr)
    else:
        interactive_mode(args.db)


if __name__ == "__main__":
    main()
