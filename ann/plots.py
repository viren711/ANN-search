"""Shared plotting helpers (matplotlib, headless)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_recall_latency(series, path, title, exact=None, x="mean_latency_ms", y="10-recall@10"):
    """series: {label: DataFrame with columns x, y}. exact: optional 1-row DataFrame."""
    fig, ax = plt.subplots(figsize=(6, 4.2))
    for label, df in series.items():
        df = df.sort_values(x)
        ax.plot(df[x], df[y], marker="o", label=label)
    if exact is not None and len(exact):
        ax.scatter(exact[x], exact[y], marker="*", s=160, c="k", zorder=5, label="FAISS exact")
    ax.set_xscale("log")
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
        ax.set_xscale("log")
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
