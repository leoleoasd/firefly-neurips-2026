#!/usr/bin/env python3
"""Parse result directory names to extract model and parser/setting names."""

from pathlib import Path

# Known models (order matters - longer names first to avoid partial matches)
MODELS = [
    "claude_45_sonnet_vlm",
    "claude_40_sonnet_vlm",
    "claude_45_sonnet",
    "claude_40_sonnet",
    "claude_37_sonnet",
    "claude_35_sonnet",
    "deepseek_r1",
    "gpt_4o_mini",
    "gpt_4o",
    "gpt_5",
    "qwen25_32b",
    "qwen25_3b",
]

# Known parsers/settings
PARSERS = [
    "no_visual_cue_numeric",
    "no_visual_cue",
    "numeric",
    "default",
]


def parse_result_dir(dir_path: str | Path) -> tuple[str, str]:
    """
    Extract model name and parser name from a result directory path.

    Args:
        dir_path: Path like '../results/webarena_lite/claude_45_sonnet_numeric'

    Returns:
        Tuple of (model_name, parser_name)

    Raises:
        ValueError: If unable to parse the directory name
    """
    dir_name = Path(dir_path).name

    # Try to match known models
    for model in MODELS:
        if dir_name.startswith(model + "_"):
            remainder = dir_name[len(model) + 1 :]
            # Check if remainder is a known parser
            if remainder in PARSERS:
                return (model, remainder)

    raise ValueError(f"Unable to parse directory name: {dir_name}")


def parse_result_dirs(dir_paths: list[str | Path]) -> list[tuple[str, str, str]]:
    """
    Parse multiple result directory paths.

    Args:
        dir_paths: List of paths

    Returns:
        List of (dir_path, model_name, parser_name) tuples
    """
    results = []
    for dir_path in dir_paths:
        model, parser = parse_result_dir(dir_path)
        results.append((str(dir_path), model, parser))
    return results


if __name__ == "__main__":
    # Example usage
    test_dirs = [
        "../results/webarena_lite/claude_45_sonnet_numeric",
        "../results/webarena_lite/claude_37_sonnet_numeric",
        "../results/webarena_lite/gpt_4o_default",
        "../results/webarena_lite/gpt_5_no_visual_cue",
        "../results/webarena_lite/claude_35_sonnet_no_visual_cue_numeric",
        "../results/webarena_lite/gpt_4o_mini_default",
    ]

    for dir_path in test_dirs:
        model, parser = parse_result_dir(dir_path)
        print(f"{dir_path} -> model={model}, parser={parser}")
