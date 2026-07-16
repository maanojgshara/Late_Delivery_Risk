"""Shared utilities: logging, progress bars and an Artifacts helper.

The Artifacts object centralises every side-effect output (figures, tables, JSON run
records, serialised models) into an organised folder tree and logs each write, so the
rest of the pipeline stays focused on logic.
"""
import os
import json
import logging
import joblib
import pandas as pd

try:
    from tqdm.auto import tqdm                      # progress bars for long loops
except Exception:                                   # graceful fallback if tqdm is absent
    class tqdm:
        """No-op fallback for tqdm when the package is unavailable.

        Supports iteration (`for x in tqdm(seq)`) and manual bars (`tqdm(total=n)` with
        update()/close()), so the pipeline runs unchanged without tqdm installed.
        Parameters: iterable optionally wraps a sequence; total/desc/etc. are ignored.
        Returns: an iterable/bar-like no-op object.
        """
        def __init__(self, iterable=None, *a, **k):
            """Store the optional iterable; bar kwargs (total/desc/...) are ignored."""
            self.iterable = iterable
        def __iter__(self):
            """Iterate over the wrapped sequence (or nothing if none was given)."""
            return iter(self.iterable or [])
        def update(self, *a, **k):
            """No-op stand-in for tqdm.update."""
            return None
        def close(self):
            """No-op stand-in for tqdm.close."""
            return None
        def __enter__(self):
            """Enter the context manager, returning this no-op bar."""
            return self
        def __exit__(self, *a):
            """Exit the context manager without suppressing exceptions."""
            return False


def get_logger(name="main", log_dir="artifacts/logs"):
    """Create a logger that writes to both the console and an artifacts log file.

    Parameters:
        name: logger name and `<name>.log` file stem.
        log_dir: directory for the log file (created if missing).
    Returns:
        logging.Logger: configured logger with duplicate handlers cleared.
    """
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S")
    fh = logging.FileHandler(os.path.join(log_dir, f"{name}.log"), mode="w")
    fh.setFormatter(fmt); logger.addHandler(fh)
    sh = logging.StreamHandler(); sh.setFormatter(fmt); logger.addHandler(sh)
    logger.propagate = False
    return logger


class Artifacts:
    """Organised output manager for figures, tables, run records and models.

    Creates artifacts/{figures,tables,logs,runs} and a models directory from the config,
    owns the run logger, and exposes save helpers that also log every write.
    Parameters: cfg is the loaded configuration dict.
    Returns: an Artifacts instance with `.logger` and save methods.
    """
    def __init__(self, cfg):
        """Create the artifacts/{figures,tables,logs,runs} + models dirs and the run logger.
        Parameters: cfg is the loaded configuration dict (paths.artifact_dir / paths.models_dir).
        Returns: None (constructor)."""
        a = cfg["paths"]["artifact_dir"]
        self.fig_dir = os.path.join(a, "figures")
        self.table_dir = os.path.join(a, "tables")
        self.log_dir = os.path.join(a, "logs")
        self.run_dir = os.path.join(a, "runs")
        self.models_dir = cfg["paths"]["models_dir"]
        for d in (self.fig_dir, self.table_dir, self.log_dir, self.run_dir, self.models_dir):
            os.makedirs(d, exist_ok=True)
        self.logger = get_logger("main", self.log_dir)

    @staticmethod
    def _dest(base, subdir):
        """Resolve (and create) the target directory, optionally inside a per-model subfolder.
        Parameters: base is the root dir; subdir is an optional sub-folder name (e.g. a model).
        Returns: the destination directory path."""
        d = os.path.join(base, subdir) if subdir else base
        os.makedirs(d, exist_ok=True)
        return d

    def fig(self, figure, name, subdir=None):
        """Save a Matplotlib figure as a tight-bbox PNG (optionally in a per-model subfolder).
        Parameters: figure is the Figure; name is the file stem; subdir routes per-model outputs.
        Returns: the written path."""
        path = os.path.join(self._dest(self.fig_dir, subdir), f"{name}.png")
        figure.savefig(path, dpi=120, bbox_inches="tight")
        self.logger.info(f"saved figure -> {path}"); return path

    def table(self, df, name, index=True, subdir=None):
        """Save a DataFrame/Series as a CSV (optionally in a per-model subfolder).
        Parameters: df is the table; name is the file stem; index toggles the row index; subdir routes per-model outputs.
        Returns: the written path."""
        path = os.path.join(self._dest(self.table_dir, subdir), f"{name}.csv")
        pd.DataFrame(df).to_csv(path, index=index)
        self.logger.info(f"saved table  -> {path}"); return path

    def run(self, payload, name, subdir=None):
        """Write a JSON run/config record for lineage (optionally in a per-model subfolder).
        Parameters: payload is a JSON-serialisable dict; name is the file stem; subdir routes per-model outputs.
        Returns: the written path."""
        path = os.path.join(self._dest(self.run_dir, subdir), f"{name}.json")
        with open(path, "w") as fh:
            json.dump(payload, fh, indent=2, default=str)
        self.logger.info(f"tracked run  -> {path}"); return path

    def model(self, obj, name, subdir=None):
        """Serialise a fitted model/pipeline with joblib (optionally in a per-model subfolder).
        Parameters: obj is the estimator/pipeline; name is the file stem; subdir routes per-model outputs.
        Returns: the written .joblib path."""
        path = os.path.join(self._dest(self.models_dir, subdir), f"{name}.joblib")
        joblib.dump(obj, path)
        self.logger.info(f"saved model  -> {path}"); return path


def label_bars(ax, fmt="{:.3f}", horizontal=False, fontsize=8, pad=3):
    """Annotate each bar in an Axes with its numeric value (above vertical / beside horizontal bars).
    Parameters: ax is the Axes; fmt formats the value; horizontal flags barh; fontsize/pad style the labels.
    Returns: the same Axes (for chaining).
    """
    for p in ax.patches:
        if horizontal:
            w = p.get_width()
            ax.annotate(fmt.format(w), (w, p.get_y() + p.get_height() / 2),
                        ha="left", va="center", fontsize=fontsize,
                        xytext=(pad, 0), textcoords="offset points")
        else:
            h = p.get_height()
            ax.annotate(fmt.format(h), (p.get_x() + p.get_width() / 2, h),
                        ha="center", va="bottom", fontsize=fontsize,
                        xytext=(0, pad), textcoords="offset points")
    return ax
