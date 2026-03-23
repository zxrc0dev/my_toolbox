import pandas as pd
import re
import shutil, zipfile, kagglehub
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
import math


def df_overview(df):
    print(f"{'='*33} Shape {'='*33}")
    print(df.shape)
    print(f"{'='*33} Info {'='*33}")
    print(df.info())
    print(f"{'='*33} Columns {'='*33}")
    print(df.columns)
    print(f"{'='*33} Describe {'='*33}")
    print(df.describe())
    print(f"{'='*33} NaN {'='*33}")
    print(df.isnull().sum())
    print(f"{'='*33} Duplicates {'='*33}")
    print(df.duplicated().sum())
    print(f"{'='*33} Cardinality & Top Values {'='*33}")
    for c in df.select_dtypes(include='object').columns:
        print(c, df[c].nunique(), df[c].value_counts(normalize=True).head())


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:

    def clean_name(name: str) -> str:
        s1 = re.sub('([a-z0-9])([A-Z])', r'\1_\2', name)
        s2 = re.sub(r'[^a-zA-Z0-9]+', '_', s1)
        return s2.lower().strip('_')

    df.columns = [clean_name(col) for col in df.columns]
    return df


def download_data(dataset_path: str, force: bool = False):
    raw_dir = Path(__file__).resolve().parents[1] / "data" / "01_raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    has_data = any(f for f in raw_dir.iterdir() if f.is_file() and not f.name.startswith('.'))

    if not force and has_data:
        print(f"Data exists in {raw_dir}. Skipping.")
        return

    print(f"Downloading {dataset_path}...")
    downloaded_path = Path(kagglehub.dataset_download(dataset_path, force_download=force))

    if downloaded_path.suffix == ".zip":
        with zipfile.ZipFile(downloaded_path, "r") as z:
            z.extractall(raw_dir)
    elif downloaded_path.is_dir():
        for f in downloaded_path.iterdir():
            if f.is_file(): shutil.copy(f, raw_dir)
    else:
        shutil.copy(downloaded_path, raw_dir)

    print(f"Files saved to: {raw_dir}")


def plots(
    data,
    features,
    kind="hist",          # "hist", "count", "bar", "box"
    value=None,           # for bar: numeric column to aggregate
    group=None,           # for box: grouping column on x-axis
    hue=None,
    kde=False,
    showfliers=False,
    palette=None,
    style="whitegrid",
    cols=3,
    figsize_per_row=(18, 5),
):
    """
    Plot multiple seaborn charts for a list of features.

    Parameters
    ----------
    data : pandas.DataFrame
        Source dataframe.
    features : list[str]
        Columns to plot, one subplot per feature.
    kind : str, default "hist"
        Plot type: "hist", "count", "bar", or "box".
    value : str, optional
        Numeric column used as y in bar plots.
        Each feature is used as x.
    group : str, optional
        Grouping column used as x in box plots.
        Each feature is used as y.
    hue : str, optional
        Seaborn hue variable.
    kde : bool, default False
        Whether to show KDE for hist plots.
    showfliers : bool, default False
        Whether to show outliers in box plots.
    palette : str or list, optional
        Seaborn palette.
    style : str, default "whitegrid"
        Seaborn style.
    cols : int, default 3
        Number of subplot columns.
    figsize_per_row : tuple[int, int], default (18, 5)
        Figure width and row height.

    Returns
    -------
    fig, axes
        Matplotlib figure and flattened axes array.
    """
    valid_kinds = {"hist", "count", "bar", "box"}
    if kind not in valid_kinds:
        raise ValueError(f"`kind` must be one of {valid_kinds}, got {kind!r}")

    if not features:
        raise ValueError("`features` must not be empty")

    missing = [col for col in features if col not in data.columns]
    if missing:
        raise ValueError(f"Columns not found in data: {missing}")

    if hue is not None and hue not in data.columns:
        raise ValueError(f"`hue` column not found: {hue}")

    if kind == "bar":
        if value is None:
            raise ValueError("`value` is required when kind='bar'")
        if value not in data.columns:
            raise ValueError(f"`value` column not found: {value}")

    if kind == "box" and group is not None and group not in data.columns:
        raise ValueError(f"`group` column not found: {group}")

    sns.set_theme(style=style)

    n = len(features)
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(figsize_per_row[0], figsize_per_row[1] * rows),
        squeeze=False
    )
    axes = axes.ravel()

    for ax, feature in zip(axes, features):
        if kind == "hist":
            sns.histplot(
                data=data,
                x=feature,
                hue=hue,
                kde=kde,
                ax=ax,
                palette=palette
            )

        elif kind == "count":
            sns.countplot(
                data=data,
                x=feature,
                hue=hue,
                ax=ax,
                palette=palette
            )
            for container in ax.containers:
                ax.bar_label(container, fmt="%d")

        elif kind == "bar":
            sns.barplot(
                data=data,
                x=feature,
                y=value,
                hue=hue,
                ax=ax,
                palette=palette
            )
            for container in ax.containers:
                ax.bar_label(container, fmt="%.2f")

        elif kind == "box":
            if group is None:
                sns.boxplot(
                    data=data,
                    x=feature,
                    ax=ax,
                    showfliers=showfliers,
                    palette=palette
                )
            else:
                sns.boxplot(
                    data=data,
                    x=group,
                    y=feature,
                    hue=hue,
                    ax=ax,
                    showfliers=showfliers,
                    palette=palette
                )

        ax.set_title(feature)
        ax.set_xlabel("")

    for ax in axes[n:]:
        ax.remove()

    plt.tight_layout()
    return fig, axes[:n]

