#!/usr/bin/env python3
"""
Script to clear everything in the SGLang registry.
"""

import ray

from shared.sglang_registry import get_or_create_registry


def main():
    """Clear all entries from the SGLang registry."""
    # Initialize Ray
    ray.init(address="auto")

    try:
        # Get the registry
        registry = get_or_create_registry("sglang_registry")

        # Show current state before clearing
        dump_before = ray.get(registry.dump.remote())
        print(f"Registry before clearing: {dump_before}")
        total_entries = sum(len(urls) for urls in dump_before.values())
        print(f"Total entries to clear: {total_entries}")

        # Clear the registry
        ray.get(registry.clear.remote())
        print("Registry cleared successfully.")

        # Verify it's cleared
        dump_after = ray.get(registry.dump.remote())
        print(f"Registry after clearing: {dump_after}")

        if dump_after:
            print("WARNING: Registry is not empty after clearing!")
        else:
            print("Registry is now empty.")

    finally:
        # Shutdown Ray
        ray.shutdown()


if __name__ == "__main__":
    main()
