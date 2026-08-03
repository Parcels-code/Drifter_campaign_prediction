import matplotlib
import matplotlib.pyplot as plt
import glob
import xarray as xr
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.collections import LineCollection
import matplotlib.tri as mtri
import numpy as np
import pandas as pd
import polars as pl
from tqdm.auto import tqdm

import parcels

DIR = "/storage/shared/oceanparcels/input_data/MatroosWaddenSea/DCSMv7_harmonie/"


def make_animation(file, time_step=np.timedelta64(30, "m"), frame_stride=6, fps=10, show_progress=True):
    df = parcels.read_particlefile(file)

    files = sorted(glob.glob(f"{DIR}/flow/dcsm_fm100m_harmonie_*"))
    ds = xr.open_dataset(files[0])
    ds = ds.isel(time=0, zc=0)
    zero_faces = (ds["U"] == 0) & (ds["V"] == 0)
    ds = ds.isel(n_face=(~zero_faces).values)

    triang = mtri.Triangulation(
        ds["Mesh_node_x"].data,
        ds["Mesh_node_y"].data,
        triangles=ds["tri_face_nodes"].data)

    t_values = df["t"].to_numpy()
    timerange = np.arange(
        t_values.min(),
        t_values.max() + time_step,
        time_step,
    )

    timerange = timerange[::frame_stride]  # default: every 3 hours for animation speed

    # Build a color map by release time (all particles released together share a color).
    release_time_by_pid = df.group_by("particle_id").agg(
        pl.col("t").min().alias("release_time")
    )
    release_times = release_time_by_pid["release_time"].unique().sort().to_list()

    colormap = matplotlib.colormaps["tab20b"]
    release_to_color = {
        rt: colormap(i / max(len(release_times) - 1, 1))
        for i, rt in enumerate(release_times)
    }
    trajectory_to_color = {
        row["particle_id"]: release_to_color[row["release_time"]]
        for row in release_time_by_pid.iter_rows(named=True)
    }

    # Precompute frame data once; this avoids scanning/filtering the dataframe at every frame.
    grouped = df.group_by("t", maintain_order=True).agg(
        [
            pl.col("x").alias("x"),
            pl.col("y").alias("y"),
            pl.col("particle_id").alias("particle_id"),
        ]
    )
    frame_lookup = {}
    for row in grouped.iter_rows(named=True):
        offsets = np.column_stack((np.asarray(row["x"]), np.asarray(row["y"])))
        colors = np.asarray([trajectory_to_color[p] for p in row["particle_id"]])
        frame_lookup[np.datetime64(row["t"], "ns")] = (offsets, colors)

    # Precompute each particle trajectory once so trail updates only slice numpy arrays.
    trajectories = {}
    for row in (
        df.sort(["particle_id", "t"])
        .group_by("particle_id", maintain_order=True)
        .agg([
            pl.col("t").alias("t"),
            pl.col("x").alias("x"),
            pl.col("y").alias("y"),
        ])
        .iter_rows(named=True)
    ):
        trajectories[row["particle_id"]] = (
            np.asarray(row["t"], dtype="datetime64[ns]"),
            np.asarray(row["x"]),
            np.asarray(row["y"]),
        )

    # figure setup
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_xlim([df["x"].min(), df["x"].max()])
    ax.set_ylim([df["y"].min(), df["y"].max()])
    ax.triplot(triang, color="k", lw=0.2, alpha=0.35)

    # --> plot first timestep
    first_frame = frame_lookup.get(np.datetime64(timerange[0], "ns"))
    if first_frame is None:
        first_offsets = np.empty((0, 2))
        first_colors = np.empty((0, 4))
    else:
        first_offsets, first_colors = first_frame

    scatter = ax.scatter(
        first_offsets[:, 0],
        first_offsets[:, 1],
        s=10,
        c=first_colors,
        zorder=10,
    )

    trail_collection = LineCollection([], linewidths=0.6, alpha=0.3)
    ax.add_collection(trail_collection)

    # Set initial title
    t_str = pd.to_datetime(timerange[0]).strftime("%Y-%m-%d %H:%M:%S")
    title = ax.set_title(f"Particles on {t_str}")

    # loop over for animation
    def animate(i):
        t_str = pd.to_datetime(timerange[i]).strftime("%Y-%m-%d %H:%M:%S")
        title.set_text(f"Particles on {t_str}")

        frame = frame_lookup.get(np.datetime64(timerange[i], "ns"))
        if frame is not None:
            offsets, colors = frame
            scatter.set_offsets(offsets)
            scatter.set_color(colors)
        else:
            scatter.set_offsets(np.empty((0, 2)))
            scatter.set_color(np.empty((0, 4)))

        trail_length = min(10, i)  # trails will have max length of 10 time steps
        if trail_length > 0:
            start_time = np.datetime64(timerange[max(0, i - trail_length)], "ns")
            end_time = np.datetime64(timerange[i], "ns")
            trail_segments = []
            trail_colors = []

            for pid, (times, xs, ys) in trajectories.items():
                start_idx = np.searchsorted(times, start_time, side="left")
                end_idx = np.searchsorted(times, end_time, side="right")
                if end_idx - start_idx > 1:
                    trail_segments.append(np.column_stack((xs[start_idx:end_idx], ys[start_idx:end_idx])))
                    trail_colors.append(trajectory_to_color[pid])

            trail_collection.set_segments(trail_segments)
            trail_collection.set_color(trail_colors)
        else:
            trail_collection.set_segments([])

        return scatter, title, trail_collection


    # Create animation
    n_frames = len(timerange)
    anim = FuncAnimation(fig, animate, frames=n_frames, interval=100, blit=True)

    if show_progress and tqdm is not None:
        with tqdm(total=n_frames, desc="Rendering GIF", unit="frame") as pbar:
            last_frame = -1

            def _progress_callback(frame_idx, _n_total):
                nonlocal last_frame
                if frame_idx > last_frame:
                    pbar.update(frame_idx - last_frame)
                    last_frame = frame_idx

            anim.save(
                file.replace(".parquet", ".gif"),
                writer=PillowWriter(fps=fps),
                progress_callback=_progress_callback,
            )
            if pbar.n < n_frames:
                pbar.update(n_frames - pbar.n)
    else:
        anim.save(file.replace(".parquet", ".gif"), writer=PillowWriter(fps=fps))
    return anim


def make_plot(file):
    df = parcels.read_particlefile(file)
    files = sorted(glob.glob(f"{DIR}/flow/dcsm_fm100m_harmonie_*"))
    ds = xr.open_dataset(files[0])

    ds = ds.isel(time=0, zc=0)
    zero_faces = (ds["U"] == 0) & (ds["V"] == 0)
    ds = ds.isel(n_face=(~zero_faces).values)

    triang = mtri.Triangulation(
        ds["Mesh_node_x"].data,
        ds["Mesh_node_y"].data,
        triangles=ds["tri_face_nodes"].data)

    # Build a color map by release time (all particles released together share a color).
    release_time_by_pid = df.group_by("particle_id").agg(
        pl.col("t").min().alias("release_time")
    )
    release_times = release_time_by_pid["release_time"].unique().sort().to_list()

    colormap = matplotlib.colormaps["tab20b"]
    release_to_color = {
        rt: colormap(i / max(len(release_times) - 1, 1))
        for i, rt in enumerate(release_times)
    }
    trajectory_to_color = {
        row["particle_id"]: release_to_color[row["release_time"]]
        for row in release_time_by_pid.iter_rows(named=True)
    }

    # figure setup
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_xlim([df["x"].min(), df["x"].max()])
    ax.set_ylim([df["y"].min(), df["y"].max()])
    ax.triplot(triang, color="k", lw=0.2, alpha=0.35)

    pids = np.array(df["particle_id"].unique())
    rng = np.random.default_rng(1636)
    rng.shuffle(pids)

    for pid in pids:
        traj = df.filter(pl.col("particle_id") == pid)
        lines = ax.plot(traj["x"], traj["y"], color=trajectory_to_color[pid], linewidth=0.6, alpha=0.3)
        ax.plot(traj["x"][-1], traj["y"][-1], marker="o", color=trajectory_to_color[pid], markersize=3)

    t_values = df["t"].to_numpy()
    t_str = pd.to_datetime(t_values.min()).strftime("%Y-%m-%d %H:%M:%S")
    t_str_end = pd.to_datetime(t_values.max()).strftime("%Y-%m-%d %H:%M:%S")
    title = ax.set_title(f"Particles between {t_str} and {t_str_end}")

    fig.savefig(file.replace(".parquet", ".png"), dpi=300)
