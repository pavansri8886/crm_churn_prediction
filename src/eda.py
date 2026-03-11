"""
src/eda.py
----------
Exploratory Data Analysis — generates and saves all plots to logs/plots/.
Called automatically by run_pipeline.py after data loading.

Plots generated:
  1.  Churn distribution (pie + bar by contract)
  2.  Numerical distributions by churn status
  3.  Correlation matrix
  4.  Churn rate by key categorical variables
  5.  Tenure segments analysis
  6.  Monthly charges box plot
  7.  Service bundle score vs churn
  8.  Payment method churn rates
  9.  High-risk combo analysis
  10. Missing data heatmap (before cleaning)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path
from loguru import logger

# ── Plot settings ─────────────────────────────────────────────────────────────
plt.rcParams["figure.figsize"] = (12, 6)
plt.rcParams["font.size"] = 11
sns.set_style("whitegrid")

COLORS       = ["#3498db", "#e74c3c"]   # blue = no churn, red = churn
PALETTE      = "RdBu_r"
PLOTS_DIR    = Path("logs/plots")


# ── Main entry point ──────────────────────────────────────────────────────────
def run_eda(df: pd.DataFrame, plots_dir: str = None) -> None:
    """
    Run all EDA plots on the raw (uncleaned) dataframe.
    Call this right after load_data(), before preprocessing.
    """
    output_dir = Path(plots_dir) if plots_dir else PLOTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("EXPLORATORY DATA ANALYSIS")
    logger.info(f"Saving plots to: {output_dir}/")
    logger.info("=" * 60)

    # Make TotalCharges numeric for plotting (don't modify original df)
    df = df.copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    _plot_missing_heatmap(df, output_dir)
    _plot_churn_distribution(df, output_dir)
    _plot_numerical_distributions(df, output_dir)
    _plot_correlation_matrix(df, output_dir)
    _plot_churn_by_categoricals(df, output_dir)
    _plot_tenure_segments(df, output_dir)
    _plot_monthly_charges_boxplot(df, output_dir)
    _plot_service_bundle(df, output_dir)
    _plot_payment_method(df, output_dir)
    _plot_high_risk_combo(df, output_dir)

    logger.success(f"EDA complete — {len(list(output_dir.glob('*.png')))} plots saved to {output_dir}/")


# ── Plot 1 — Missing Data Heatmap ─────────────────────────────────────────────
def _plot_missing_heatmap(df: pd.DataFrame, out: Path) -> None:
    """Visualise missing values across the dataset (before cleaning)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: heatmap of missing values
    missing = df.isnull()
    if missing.any().any():
        sns.heatmap(missing, cbar=False, yticklabels=False,
                    cmap=["#2ecc71", "#e74c3c"], ax=axes[0])
        axes[0].set_title("Missing Values Heatmap\n(red = missing)", fontweight="bold")
    else:
        axes[0].text(0.5, 0.5, "No missing values\nin standard columns",
                     ha="center", va="center", fontsize=14, transform=axes[0].transAxes)
        axes[0].set_title("Missing Values Heatmap", fontweight="bold")
        axes[0].axis("off")

    # Right: bar chart of missing counts per column
    missing_counts = df.isnull().sum().sort_values(ascending=False)
    missing_counts = missing_counts[missing_counts > 0]
    if len(missing_counts) > 0:
        missing_counts.plot(kind="bar", ax=axes[1], color="#e74c3c", edgecolor="white")
        axes[1].set_title("Missing Value Counts per Column", fontweight="bold")
        axes[1].set_ylabel("Number of missing values")
        axes[1].tick_params(axis="x", rotation=45)
    else:
        # TotalCharges whitespace entries — show them explicitly
        whitespace = (df["TotalCharges"].astype(str).str.strip() == "").sum() if "TotalCharges" in df.columns else 0
        axes[1].bar(["TotalCharges\n(whitespace)"], [whitespace], color="#f39c12", edgecolor="white")
        axes[1].set_title("Hidden Missing Values\n(whitespace strings)", fontweight="bold")
        axes[1].set_ylabel("Count")

    plt.suptitle("Data Quality — Missing Values Audit", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    _save(fig, out / "01_missing_values.png")


# ── Plot 2 — Churn Distribution ───────────────────────────────────────────────
def _plot_churn_distribution(df: pd.DataFrame, out: Path) -> None:
    """Pie chart + churn by contract type."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Pie chart
    churn_counts = df["Churn"].value_counts()
    axes[0].pie(
        churn_counts.values,
        labels=["No Churn", "Churn"],
        colors=COLORS,
        autopct="%1.1f%%",
        startangle=90,
        explode=(0, 0.05),
        shadow=True,
    )
    axes[0].set_title("Overall Churn Distribution", fontsize=13, fontweight="bold")

    # Churn by contract type
    contract_churn = df.groupby(["Contract", "Churn"]).size().unstack(fill_value=0)
    contract_churn.plot(kind="bar", ax=axes[1], color=COLORS, edgecolor="white", width=0.7)
    axes[1].set_title("Churn Count by Contract Type", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Contract Type")
    axes[1].set_ylabel("Number of Customers")
    axes[1].tick_params(axis="x", rotation=15)
    axes[1].legend(["No Churn", "Churn"])

    plt.suptitle("Target Variable Analysis — Customer Churn", fontsize=14, fontweight="bold")
    plt.tight_layout()
    _save(fig, out / "02_churn_distribution.png")


# ── Plot 3 — Numerical Distributions ─────────────────────────────────────────
def _plot_numerical_distributions(df: pd.DataFrame, out: Path) -> None:
    """Overlapping histograms: churned vs non-churned for each numerical column."""
    num_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    titles   = ["Tenure (months)", "Monthly Charges (€)", "Total Charges (€)"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for ax, col, title in zip(axes, num_cols, titles):
        if col not in df.columns:
            continue
        no_churn = df[df["Churn"] == "No"][col].dropna()
        churned  = df[df["Churn"] == "Yes"][col].dropna()

        ax.hist(no_churn, bins=30, alpha=0.6, color=COLORS[0], label="No Churn", density=True)
        ax.hist(churned,  bins=30, alpha=0.6, color=COLORS[1], label="Churn",    density=True)

        # Add vertical median lines
        ax.axvline(no_churn.median(), color=COLORS[0], linestyle="--", linewidth=1.5,
                   label=f"Median: {no_churn.median():.0f}")
        ax.axvline(churned.median(),  color=COLORS[1], linestyle="--", linewidth=1.5,
                   label=f"Median: {churned.median():.0f}")

        ax.set_title(title, fontweight="bold")
        ax.set_ylabel("Density")
        ax.legend(fontsize=9)

    plt.suptitle("Numerical Variable Distributions by Churn Status",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    _save(fig, out / "03_numerical_distributions.png")


# ── Plot 4 — Correlation Matrix ───────────────────────────────────────────────
def _plot_correlation_matrix(df: pd.DataFrame, out: Path) -> None:
    """Heatmap of correlations between numerical columns + churn."""
    df_num = df[["tenure", "MonthlyCharges", "TotalCharges"]].copy()
    df_num["Churn"] = (df["Churn"] == "Yes").astype(int)

    fig, ax = plt.subplots(figsize=(8, 6))
    mask = np.triu(np.ones_like(df_num.corr(), dtype=bool))  # upper triangle mask

    sns.heatmap(
        df_num.corr(),
        annot=True,
        fmt=".2f",
        cmap=PALETTE,
        center=0,
        square=True,
        linewidths=0.5,
        mask=mask,
        ax=ax,
        annot_kws={"size": 13, "weight": "bold"},
    )
    ax.set_title("Correlation Matrix — Numerical Variables",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save(fig, out / "04_correlation_matrix.png")


# ── Plot 5 — Churn Rate by Categorical Variables ──────────────────────────────
def _plot_churn_by_categoricals(df: pd.DataFrame, out: Path) -> None:
    """Horizontal bar charts showing churn % for each categorical variable."""
    cat_cols = ["InternetService", "Contract", "PaymentMethod", "SeniorCitizen"]
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.flatten()

    for ax, col in zip(axes, cat_cols):
        if col not in df.columns:
            continue
        churn_rate = (
            df.groupby(col)["Churn"]
            .apply(lambda x: (x == "Yes").mean() * 100)
            .sort_values(ascending=True)
        )
        bars = ax.barh(churn_rate.index.astype(str), churn_rate.values,
                       color="#e74c3c", edgecolor="white", height=0.6)
        ax.set_title(f"Churn Rate by {col}", fontweight="bold")
        ax.set_xlabel("Churn Rate (%)")
        ax.set_xlim(0, max(churn_rate.values) * 1.25)

        for bar, val in zip(bars, churn_rate.values):
            ax.text(val + 0.3, bar.get_y() + bar.get_height() / 2,
                    f"{val:.1f}%", va="center", fontsize=10, fontweight="bold")

    plt.suptitle("Churn Rate by Key Categorical Variables",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    _save(fig, out / "05_churn_by_categoricals.png")


# ── Plot 6 — Tenure Segments ──────────────────────────────────────────────────
def _plot_tenure_segments(df: pd.DataFrame, out: Path) -> None:
    """Segment customers by tenure and show churn rate per segment."""
    df_temp = df.copy()
    df_temp["tenure_segment"] = pd.cut(
        df_temp["tenure"],
        bins=[0, 6, 12, 24, 48, 72],
        labels=["0-6m (New)", "6-12m", "12-24m", "24-48m", "48-72m (Loyal)"],
    )

    churn_by_segment = (
        df_temp.groupby("tenure_segment", observed=True)["Churn"]
        .apply(lambda x: (x == "Yes").mean() * 100)
    )
    segment_counts = df_temp["tenure_segment"].value_counts().sort_index()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: churn rate per segment
    colors_seg = ["#e74c3c" if v > 30 else "#f39c12" if v > 20 else "#2ecc71"
                  for v in churn_by_segment.values]
    axes[0].bar(churn_by_segment.index.astype(str), churn_by_segment.values,
                color=colors_seg, edgecolor="white")
    axes[0].set_title("Churn Rate by Tenure Segment", fontweight="bold")
    axes[0].set_ylabel("Churn Rate (%)")
    axes[0].tick_params(axis="x", rotation=15)
    for i, v in enumerate(churn_by_segment.values):
        axes[0].text(i, v + 0.5, f"{v:.1f}%", ha="center", fontweight="bold")

    # Right: customer count per segment
    axes[1].bar(segment_counts.index.astype(str), segment_counts.values,
                color="#3498db", edgecolor="white")
    axes[1].set_title("Customer Count by Tenure Segment", fontweight="bold")
    axes[1].set_ylabel("Number of Customers")
    axes[1].tick_params(axis="x", rotation=15)

    plt.suptitle("Tenure Segment Analysis", fontsize=14, fontweight="bold")
    plt.tight_layout()
    _save(fig, out / "06_tenure_segments.png")


# ── Plot 7 — Monthly Charges Box Plot ────────────────────────────────────────
def _plot_monthly_charges_boxplot(df: pd.DataFrame, out: Path) -> None:
    """Box plot of monthly charges by contract type and churn status."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: charges by churn status
    df.boxplot(column="MonthlyCharges", by="Churn", ax=axes[0],
               boxprops=dict(color="#2c3e50"),
               medianprops=dict(color="#e74c3c", linewidth=2))
    axes[0].set_title("Monthly Charges by Churn Status", fontweight="bold")
    axes[0].set_xlabel("Churn (No / Yes)")
    axes[0].set_ylabel("Monthly Charges (€)")
    plt.sca(axes[0])
    plt.title("Monthly Charges by Churn Status", fontweight="bold")

    # Right: charges by contract type
    df.boxplot(column="MonthlyCharges", by="Contract", ax=axes[1],
               boxprops=dict(color="#2c3e50"),
               medianprops=dict(color="#3498db", linewidth=2))
    axes[1].set_title("Monthly Charges by Contract Type", fontweight="bold")
    axes[1].set_xlabel("Contract Type")
    axes[1].set_ylabel("Monthly Charges (€)")
    axes[1].tick_params(axis="x", rotation=15)
    plt.sca(axes[1])
    plt.title("Monthly Charges by Contract Type", fontweight="bold")

    plt.suptitle("Monthly Charges Distribution — Outlier & Spread Analysis",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    _save(fig, out / "07_monthly_charges_boxplot.png")


# ── Plot 8 — Service Bundle Score ────────────────────────────────────────────
def _plot_service_bundle(df: pd.DataFrame, out: Path) -> None:
    """
    Show how the number of active services (bundle score) correlates with churn.
    More services = more switching cost = lower churn.
    """
    service_cols = [
        "PhoneService", "OnlineSecurity", "OnlineBackup",
        "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    ]
    df_temp = df.copy()
    df_temp["ServiceBundleScore"] = df_temp[service_cols].apply(
        lambda x: (x == "Yes").sum(), axis=1
    )
    churn_by_bundle = (
        df_temp.groupby("ServiceBundleScore")["Churn"]
        .apply(lambda x: (x == "Yes").mean() * 100)
    )
    count_by_bundle = df_temp["ServiceBundleScore"].value_counts().sort_index()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(churn_by_bundle.index, churn_by_bundle.values,
                 "o-", color="#e74c3c", linewidth=2.5, markersize=8)
    axes[0].fill_between(churn_by_bundle.index, churn_by_bundle.values,
                         alpha=0.15, color="#e74c3c")
    axes[0].set_title("Churn Rate by Service Bundle Score", fontweight="bold")
    axes[0].set_xlabel("Number of Active Services")
    axes[0].set_ylabel("Churn Rate (%)")
    axes[0].set_xticks(churn_by_bundle.index)

    axes[1].bar(count_by_bundle.index, count_by_bundle.values,
                color="#3498db", edgecolor="white")
    axes[1].set_title("Customer Count by Service Bundle Score", fontweight="bold")
    axes[1].set_xlabel("Number of Active Services")
    axes[1].set_ylabel("Number of Customers")

    plt.suptitle("Feature Engineering Validation — Service Bundle Score",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    _save(fig, out / "08_service_bundle_score.png")


# ── Plot 9 — Payment Method ───────────────────────────────────────────────────
def _plot_payment_method(df: pd.DataFrame, out: Path) -> None:
    """Stacked bar showing absolute churn counts by payment method."""
    pm_churn = df.groupby(["PaymentMethod", "Churn"]).size().unstack(fill_value=0)

    fig, ax = plt.subplots(figsize=(10, 5))
    pm_churn.plot(kind="bar", stacked=False, ax=ax,
                  color=COLORS, edgecolor="white", width=0.7)
    ax.set_title("Churn Count by Payment Method", fontsize=13, fontweight="bold")
    ax.set_xlabel("Payment Method")
    ax.set_ylabel("Number of Customers")
    ax.tick_params(axis="x", rotation=20)
    ax.legend(["No Churn", "Churn"])

    # Add churn rate annotation above each churn bar
    totals = pm_churn.sum(axis=1)
    churn_col = "Yes" if "Yes" in pm_churn.columns else 1
    no_churn_col = "No" if "No" in pm_churn.columns else 0
    for i, (idx, row) in enumerate(pm_churn.iterrows()):
        rate = row.get(churn_col, 0) / totals[idx] * 100
        ax.text(i + 0.18, row.get(churn_col, 0) + 5,
                f"{rate:.0f}%", ha="center", fontsize=10,
                color="#e74c3c", fontweight="bold")

    plt.tight_layout()
    _save(fig, out / "09_payment_method_churn.png")


# ── Plot 10 — High-Risk Combo ─────────────────────────────────────────────────
def _plot_high_risk_combo(df: pd.DataFrame, out: Path) -> None:
    """
    Validate the HighRiskCombo engineered feature:
    Month-to-month + Fiber optic + No OnlineSecurity.
    """
    df_temp = df.copy()
    df_temp["HighRiskCombo"] = (
        (df_temp["Contract"] == "Month-to-month") &
        (df_temp["InternetService"] == "Fiber optic") &
        (df_temp["OnlineSecurity"] == "No")
    ).astype(int)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: churn rate for high-risk vs others
    churn_rate = (
        df_temp.groupby("HighRiskCombo")["Churn"]
        .apply(lambda x: (x == "Yes").mean() * 100)
    )
    labels = ["Standard Risk", "High Risk Combo"]
    bar_colors = ["#3498db", "#e74c3c"]
    bars = axes[0].bar(labels, churn_rate.values, color=bar_colors, edgecolor="white", width=0.5)
    axes[0].set_title("Churn Rate: High-Risk vs Standard Customers",
                      fontweight="bold")
    axes[0].set_ylabel("Churn Rate (%)")
    axes[0].set_ylim(0, 100)
    for bar, val in zip(bars, churn_rate.values):
        axes[0].text(bar.get_x() + bar.get_width() / 2, val + 1,
                     f"{val:.1f}%", ha="center", fontsize=13, fontweight="bold")

    # Right: population breakdown
    counts = df_temp["HighRiskCombo"].value_counts()
    axes[1].pie(
        [counts.get(0, 0), counts.get(1, 0)],
        labels=["Standard Risk", "High Risk Combo"],
        colors=["#3498db", "#e74c3c"],
        autopct="%1.1f%%",
        startangle=90,
    )
    axes[1].set_title("Customer Breakdown\nHigh-Risk Combo vs Standard", fontweight="bold")

    plt.suptitle(
        "Engineered Feature Validation — HighRiskCombo\n"
        "(Month-to-month + Fiber Optic + No Online Security)",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    _save(fig, out / "10_high_risk_combo.png")


# ── Helper ────────────────────────────────────────────────────────────────────
def _save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved: {path.name}")
