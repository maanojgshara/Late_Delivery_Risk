"""Configuration loading for the pipeline.

Reads the YAML config file and applies optional command-line / environment overrides,
giving a single source of truth for paths, the target, split, preprocessing thresholds,
search budget, calibration and the candidate-model grids.
"""
import os
import yaml

DEFAULT_CONFIG_PATH = "config.yaml"


def load_config(path=None, overrides=None):
    """Load the YAML pipeline configuration and apply optional top-level overrides.

    Parameters:
        path: path to the YAML file; falls back to the PIPELINE_CONFIG env var then config.yaml.
        overrides: optional dict of top-level key -> value (e.g. from argparse); None values ignored.
    Returns:
        dict: the merged configuration.
    """
    path = path or os.environ.get("PIPELINE_CONFIG", DEFAULT_CONFIG_PATH)
    with open(path, "r") as fh:
        cfg = yaml.safe_load(fh)
    if overrides:
        cfg.update({k: v for k, v in overrides.items() if v is not None})
    return cfg
