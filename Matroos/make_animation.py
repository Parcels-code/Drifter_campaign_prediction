import matplotlib
import matplotlib.pyplot as plt
import time
import glob
import xarray as xr
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.collections import LineCollection
import matplotlib.tri as mtri
import numpy as np
import pandas as pd
import polars as pl
from tqdm.auto import tqdm

import copernicusmarine

import parcels

DIR = "/storage/shared/oceanparcels/input_data/MatroosWaddenSea/DCSMv7_harmonie/"


def make_animation(file, time_step=np.timedelta64(30, "m"), frame_stride=6, fps=10, show_progress=True):
    df = parcels.read_particlefile(file)

    fig, ax = setup_plot(
        xlim=[df["x"].min(), df["x"].max()],
        ylim=[df["y"].min(), df["y"].max()]
    )

    tic = time.time()
    t_values = df["t"].to_numpy()
    timerange = np.arange(
        t_values.min(),
        t_values.max() + time_step,
        time_step,
    )

    trange_stride = timerange[::frame_stride]  # default: every 3 hours for animation speed

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

    trails = LineCollection([], linewidths=0.6, alpha=0.3)
    ax.add_collection(trails)

    # Convert the trajectory data to a time-by-particle array.
    particle_ids = df["particle_id"].unique().to_list()
    x = np.full((len(particle_ids), len(timerange)), np.nan)
    y = np.full((len(particle_ids), len(timerange)), np.nan)

    traj = df.with_columns(
        pl.col("particle_id")
        .replace(particle_ids, range(len(particle_ids)))
        .alias("p_idx"),
        pl.int_range(pl.len())
        .alias("row_idx"),
    )

    # Map each observation to its nearest animation time index.
    traj = traj.with_columns(
        pl.Series("t_idx", np.searchsorted(timerange, traj["t"].to_numpy())).alias("t_idx")
    )

    x[traj["p_idx"].to_numpy(), traj["t_idx"].to_numpy()] = traj["x"].to_numpy()
    y[traj["p_idx"].to_numpy(), traj["t_idx"].to_numpy()] = traj["y"].to_numpy()

    # Precompute colors for each particle column.
    colors = np.asarray([trajectory_to_color[pid] for pid in particle_ids])

    # Plot first timestep
    scatter = ax.scatter(x[:, 0], y[:, 0], s=10, c=colors, zorder=10)

    # Set initial title
    t_str = pd.to_datetime(timerange[0]).strftime("%Y-%m-%d %H:%M:%S")
    title = ax.set_title(f"Particles on {t_str}")

    # loop over for animation
    def animate(i):
        t_str = pd.to_datetime(trange_stride[i]).strftime("%Y-%m-%d %H:%M:%S")
        title.set_text(f"Particles on {t_str}")

        I = np.where(timerange == trange_stride[i])[0][0]
        scatter.set_offsets(np.column_stack((x[:, I], y[:, I])))

        trail_length = min(10, I)  # trails will have max length of 10 time steps
        if trail_length > 0:
            start = max(0, I - trail_length)
            x_slice = x[:, start : I + 1]
            y_slice = y[:, start : I + 1]
            trails.set_segments(np.dstack((x_slice, y_slice)))
            trails.set_color(colors)
        else:
            trails.set_segments([])
            trails.set_color([])


    # Create animation
    n_frames = len(trange_stride)
    anim = FuncAnimation(fig, animate, frames=n_frames, interval=100)

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

    print(f"making animation took {time.time() - tic:.2f} seconds")


def make_plot(file):
    df = parcels.read_particlefile(file)

    fig, ax = setup_plot(
        xlim=[df["x"].min(), df["x"].max()],
        ylim=[df["y"].min(), df["y"].max()]
    )

    tic = time.time()
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

    pids = np.array(df["particle_id"].unique())
    rng = np.random.default_rng(1636)
    rng.shuffle(pids)

    for pid in pids:
        traj = df.filter(pl.col("particle_id") == pid)
        lines = ax.plot(traj["x"], traj["y"], color=trajectory_to_color[pid], linewidth=0.6, alpha=0.3)
        ax.plot(traj["x"][-1], traj["y"][-1], marker="o", color=trajectory_to_color[pid], markersize=3, zorder=4)

    t_values = df["t"].to_numpy()
    t_str = pd.to_datetime(t_values.min()).strftime("%Y-%m-%d %H:%M:%S")
    t_str_end = pd.to_datetime(t_values.max()).strftime("%Y-%m-%d %H:%M:%S")
    title = ax.set_title(f"Particles between {t_str} and {t_str_end}")

    fig.savefig(file.replace(".parquet", ".png"), dpi=300)
    print(f"make_plot took {time.time() - tic:.2f} seconds")


def setup_plot(xlim, ylim):
    tic = time.time()
    ds = xr.open_dataset(glob.glob(f"{DIR}/flow/dcsm_fm100m_harmonie_*")[0]).isel(time=0, zc=0)
    zero_faces = (ds["U"] == 0) & (ds["V"] == 0)
    ds = ds.isel(n_face=(~zero_faces).values)

    triang = mtri.Triangulation(
        ds["Mesh_node_x"].data,
        ds["Mesh_node_y"].data,
        triangles=ds["tri_face_nodes"].data)

    ds_waves = xr.open_dataset(glob.glob(f"{DIR}/waves/swan_kuststrook_harmonie_*.nc")[0])

    ds_wind = copernicusmarine.open_dataset(
        dataset_id="cmems_obs-wind_glo_phy_my_l4_0.125deg_PT1H",
        variables=["eastward_wind"],
        minimum_longitude=1,
        maximum_longitude=8,
        minimum_latitude=51,
        maximum_latitude=55,
        start_datetime="2024-01-01T00:00:00Z",
        end_datetime="2024-01-01T00:00:00Z",
    )

    # figure setup
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    for j in range(ds_waves["lon"].values.shape[1]):
        label = "Waves mesh" if j == 0 else None
        ax.plot(ds_waves["lon"].values[:, j], ds_waves["lat"].values[:, j], color="b", lw=0.3, alpha=0.5, label=label)
    for i in range(ds_waves["lon"].values.shape[0]):
        ax.plot(ds_waves["lon"].values[i, :], ds_waves["lat"].values[i, :], color="b", lw=0.3, alpha=0.5)

    for j in range(ds_wind["latitude"].values.shape[0]):
        label = "Wind mesh" if j == 0 else None
        ax.hlines(ds_wind["latitude"].values[j], xmin=ds_wind["longitude"].values.min(), xmax=ds_wind["longitude"].values.max(), color="r", lw=0.3, alpha=0.5, label=label, zorder=0)
    for i in range(ds_wind["longitude"].values.shape[0]):
        ax.vlines(ds_wind["longitude"].values[i], ymin=ds_wind["latitude"].values.min(), ymax=ds_wind["latitude"].values.max(), color="r", lw=0.3, alpha=0.5, zorder=1)

    ax.triplot(triang, color="k", lw=0.3, alpha=0.5, label="Flow mesh", zorder=2)

    ax.legend(loc="lower right", fontsize=8)
    ax.set_aspect("equal", adjustable="box")

    print(f"setup_plot took {time.time() - tic:.2f} seconds")

    return fig, ax
