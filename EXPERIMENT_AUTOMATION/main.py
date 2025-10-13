# main.py
"""
CLI entrypoint to run a reader-flow experiment.

Usage
-----
python -m main --config path/to/config.yaml
"""

from __future__ import annotations

import argparse
import sys

import yaml
from dotenv import load_dotenv

from orchestrator.runner import run_experiment
from helper.hooks import (
    start_docker,
    stop_docker,
)

# Load environment from .env early (SSH/JMeter creds, etc.)
load_dotenv()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Run reader-flow experiment.")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    return parser.parse_args(argv)


def merge_configs(base_config: dict, extension_config: dict) -> dict:
    """Führt Basis- und Erweiterungskonfiguration zusammen."""
    merged = base_config.copy()

    def deep_merge(source, destination):
        for key, value in source.items():
            if key in destination and isinstance(destination[key], dict) and isinstance(value, dict):
                deep_merge(value, destination[key])
            else:
                destination[key] = value
        return destination

    return deep_merge(extension_config, merged)

def load_config(path: str) -> dict:
    """Lädt YAML-Konfiguration von der Festplatte mit Unterstützung für Erweiterungen."""
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Prüfe auf extends-Attribut
    if "extends" in config:
        base_path = config.pop("extends")
        # Lade die Basiskonfiguration
        with open(base_path, "r", encoding="utf-8") as f:
            base_config = yaml.safe_load(f)
        # Führe die Konfigurationen zusammen
        return merge_configs(base_config, config)

    return config


def dispatch(config: dict) -> None:
    """
    Dispatch to the appropriate experiment flow based on config['experiment']['type'].
    """
    experiment_type = config["experiment"]["type"]

    match experiment_type:
        case "baseline_idle_no_tools" | "baseline_idle":
            run_experiment(config, experiment_type=experiment_type)

        case "spring_docker_none" | "spring_docker_idle" | "spring_docker_tools" | "spring_docker_kepler"  | "spring_docker_scaphandre" | "spring_docker_otjae"  | "spring_docker_joularjx":
            run_experiment(
                config,
                experiment_type=experiment_type,
                before_experiment_hook=start_docker,
                after_experiment_hook=stop_docker,
            )

        case _:
            raise ValueError(f"Unknown experiment type: {experiment_type}")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = load_config(args.config)
    dispatch(config)


if __name__ == "__main__":
    main(sys.argv[1:])
