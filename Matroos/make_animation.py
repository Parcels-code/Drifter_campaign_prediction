import matplotlib
import matplotlib.pyplot as plt
import glob
import xarray as xr
from matplotlib.animation import FuncAnimation, PillowWriter
import matplotlib.tri as mtri
import numpy as np
import pandas as pd
import polars as pl

import parcels

DIR = "/storage/shared/oceanparcels/input_data/MatroosWaddenSea/DCSMv7_harmonie/"

def make_animation(file, time_step=np.timedelta64(30, "m")):
    df = parcels.read_particlefile(file)

#%% Open flow files
    files = sorted(glob.glob(f"{DIR}/flow/dcsm_fm100m_harmonie_*"))
    ds = xr.open_dataset(files[0])
    ds = ds.isel(time=0, zc=0)
    zero_faces = (ds["U"] == 0) & (ds["V"] == 0)
    ds = ds.isel(n_face=(~zero_faces).values)

    triang = mtri.Triangulation(
        ds["Mesh_node_x"].data,
        ds["Mesh_node_y"].data,
        triangles=ds["tri_face_nodes"].data)

    timerange = np.arange(
        np.nanmin(df["t"]),
        np.nanmax(df["t"]) + time_step,
        time_step,
    )

    timerange = timerange[::6]  # limit to every 3 hours for animation speed

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

    # --> plot first timestep
    particles = df.filter(pl.col("t") == pl.lit(timerange[0]))
    scatter = ax.scatter(
        particles["x"],
        particles["y"],
        s=10,
        c=[trajectory_to_color[p] for p in particles["particle_id"]],
        zorder=10,
    )

    # --> initialize trails
    trail_plot = []

    # Set initial title
    t_str = pd.to_datetime(timerange[0]).strftime("%Y-%m-%d %H:%M:%S")
    title = ax.set_title(f"Particles on {t_str}")

    pids = np.array(df["particle_id"].unique())
    rng = np.random.default_rng(1636)
    rng.shuffle(pids)

    # loop over for animation
    def animate(i):
        t_str = pd.to_datetime(timerange[i]).strftime("%Y-%m-%d %H:%M:%S")
        title.set_text(f"Particles on {t_str}")

        # Find particles at current time
        particles = df.filter(pl.col("t") == pl.lit(timerange[i]))

        if len(particles) > 0:
            scatter.set_offsets(np.c_[particles["x"], particles["y"]])
            scatter.set_color([trajectory_to_color[p] for p in particles["particle_id"]])

            # --> reset trails
            for trail in trail_plot:
                trail.remove()
            trail_plot.clear()
            trail_length = min(10, i)  # trails will have max length of 10 time steps
            if trail_length > 0:
                for traj in pids:
                    traj_trail = df.filter(
                        (pl.col("particle_id") == traj)
                        & (pl.col("t") >= pl.lit(timerange[max(0, i - trail_length)]))
                        & (pl.col("t") <= pl.lit(timerange[i]))
                    )
                    if len(traj_trail) > 1:
                        (trail,) = ax.plot(
                            traj_trail["x"],
                            traj_trail["y"],
                            color=trajectory_to_color[traj],
                            linewidth=0.6,
                            alpha=0.3,
                        )
                        trail_plot.append(trail)
        else:
            scatter.set_offsets(np.empty((0, 2)))


    # Create animation
    anim = FuncAnimation(fig, animate, frames=len(timerange), interval=100)
    anim.save(file.replace(".parquet", ".gif"), writer=PillowWriter(fps=10))
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
    fig.savefig(file.replace(".parquet", ".png"), dpi=300)
