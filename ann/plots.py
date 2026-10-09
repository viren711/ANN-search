"""Shared plotting helpers (matplotlib, headless)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import ticker


def _log_x(ax):
    """Log x-axis with readable plain-number ticks at 1-2-5 steps (no overlapping 10^x labels)."""
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(ticker.LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_minor_formatter(ticker.NullFormatter())


def plot_recall_latency(series, path, title, exact=None, x="mean_latency_ms", y="10-recall@10"):
    """series: {label: DataFrame with columns x, y}. exact: optional 1-row DataFrame."""
    fig, ax = plt.subplots(figsize=(6, 4.2))
    for label, df in series.items():
        df = df.sort_values(x)
        ax.plot(df[x], df[y], marker="o", label=label)
    if exact is not None and len(exact):
        ax.scatter(exact[x], exact[y], marker="*", s=160, c="k", zorder=5, label="FAISS exact")
    _log_x(ax)
    ax.set_xlabel("mean latency / query (ms, log)" if x == "mean_latency_ms" else x)
    ax.set_ylabel(y)
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_xy(series, path, title, x, y, logx=False):
    fig, ax = plt.subplots(figsize=(6, 4.2))
    for label, df in series.items():
        df = df.sort_values(x)
        ax.plot(df[x], df[y], marker="o", label=label)
    if logx:
        _log_x(ax)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
