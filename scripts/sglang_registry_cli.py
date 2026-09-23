#!/usr/bin/env python3
"""
CLI for the shared SGLang registry (Ray detached actor).

Usage:
    python scripts/sglang_registry_cli.py dump
    python scripts/sglang_registry_cli.py add <key> <url>
    python scripts/sglang_registry_cli.py remove <key>            # remove all URLs for key
    python scripts/sglang_registry_cli.py remove <key> --url <u>  # remove one URL
    python scripts/sglang_registry_cli.py clear
"""

import argparse
import json
import sys

import ray

from shared.sglang_registry import get_or_create_registry


def cmd_dump(registry, _args):
    data = ray.get(registry.dump.remote())
    if not data:
        print("(empty)")
        return
    print(json.dumps(data, indent=2))


def cmd_add(registry, args):
    ray.get(registry.add.remote(args.key, args.url))
    print(f"Added {args.url!r} to key {args.key!r}")


def cmd_remove(registry, args):
    if args.url:
        removed = ray.get(registry.remove.remote(args.key, args.url))
        if removed:
            print(f"Removed {args.url!r} from key {args.key!r}")
        else:
            print(f"URL {args.url!r} not found under key {args.key!r}")
            sys.exit(1)
    else:
        urls = ray.get(registry.remove_key.remote(args.key))
        if urls:
            print(f"Removed key {args.key!r} ({len(urls)} URL(s))")
        else:
            print(f"Key {args.key!r} not found")
            sys.exit(1)


def cmd_clear(registry, _args):
    dump_before = ray.get(registry.dump.remote())
    total = sum(len(urls) for urls in dump_before.values())
    ray.get(registry.clear.remote())
    print(f"Cleared {len(dump_before)} key(s), {total} URL(s)")


def main():
    parser = argparse.ArgumentParser(
        description="Operate the shared SGLang registry",
    )
    parser.add_argument(
        "--registry-name",
        default="sglang_registry",
        help="Name of the Ray actor (default: sglang_registry)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # dump
    sub.add_parser("dump", help="Dump all registry entries as JSON")

    # add
    p_add = sub.add_parser("add", help="Register a URL under a key")
    p_add.add_argument("key", help="Registry key (e.g. rm_worker)")
    p_add.add_argument("url", help="URL to register")

    # remove
    p_rm = sub.add_parser("remove", help="Remove all URLs for a key, or a single URL")
    p_rm.add_argument("key", help="Registry key")
    p_rm.add_argument("--url", default=None, help="Specific URL to remove (omit to remove entire key)")

    # clear
    sub.add_parser("clear", help="Clear all entries from the registry")

    args = parser.parse_args()

    ray.init(address="auto")
    try:
        registry = get_or_create_registry(args.registry_name)
        handler = {
            "dump": cmd_dump,
            "add": cmd_add,
            "remove": cmd_remove,
            "clear": cmd_clear,
        }[args.command]
        handler(registry, args)
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
