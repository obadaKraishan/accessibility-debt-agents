"""
App identity for AndroidControl episodes.

AndroidControl records which app an episode ran in only on `open_app`
actions. Every other step stores an empty `app_name`, and most episodes do
not begin with `open_app`, so the action record cannot cluster or adjust by
app on its own.

The accessibility forest can. A resource id is written `<package>:id/<name>`,
so the package that dominates an episode's resource ids names the app the
demonstration ran in. Platform packages that appear on every device (status
bar, keyboard, launcher, permission dialogs) are ignored; the list lives in
config.yaml under androidcontrol.system_packages.

This is used for clustering and adjustment only. It never enters the
codebook, so Rico and AndroidControl are still classified from identical
information.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd
import pyarrow.parquet as pq

__all__ = ["androidcontrol_episode_apps"]

_PACKAGE = r"^([A-Za-z0-9_.]+):id/"


def androidcontrol_episode_apps(
    nodes_path: Path, system_packages: Iterable[str]
) -> pd.DataFrame:
    """
    One row per episode that has at least one non-platform resource id.

    Columns: episode_id, app (the dominant package), app_share (the share of
    the episode's non-platform resource ids carrying that package). Episodes
    with no usable resource id are absent; the caller decides how to treat
    them and should report how many there are.
    """
    nodes = pq.read_table(nodes_path, columns=["episode_id", "resource_id"]).to_pandas()
    nodes["app"] = nodes["resource_id"].fillna("").str.extract(_PACKAGE, expand=False)
    nodes = nodes[nodes["app"].notna() & ~nodes["app"].isin(set(system_packages))]
    counts = (nodes.groupby(["episode_id", "app"]).size().rename("n").reset_index()
                   .sort_values(["episode_id", "n", "app"], ascending=[True, False, True]))
    totals = counts.groupby("episode_id")["n"].sum()
    top = counts.drop_duplicates("episode_id").set_index("episode_id")
    top["app_share"] = top["n"] / totals
    return top[["app", "app_share"]].reset_index()
