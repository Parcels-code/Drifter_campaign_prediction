import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

import matplotlib.tri as mtri
import uxarray as ux

import parcels

#%% Open files
files = "/storage/shared/oceanparcels/input_data/MatroosWaddenSea/DCSMv7_harmoby/maps2d_dcsm7_harmonie_2025*.nc"

def drop_analysis_time(d):
    # works whether analysis_time is a coord or variable
    return d.drop_vars("analysis_time", errors="ignore")

ds = xr.open_mfdataset(
    files,
    preprocess=drop_analysis_time,
    combine="nested",      # keep file order; do not auto-align by coords
    concat_dim="time",     # concatenate along time
    coords="minimal",      # keep only needed coords
    data_vars="minimal",
    join="override",       # use indexes from first file (same-size requirement)
    compat="override",     # do not fail on conflicting non-concat vars/coords
)

#%% Make all faces triangular (for uxarray/parcels compatibility)
node_x = ds["Mesh_node_x"].values
node_y = ds["Mesh_node_y"].values
raw_face_nodes = ds["Mesh_face_nodes"].values

# Keep original mesh connectivity (0-based, padded with -1) for uxgrid/parcels.
orig_face_nodes = np.where(np.isfinite(raw_face_nodes), raw_face_nodes - 1, -1).astype("int64")

# Use integer-safe connectivity for triangulation logic (1-based with 0 as invalid).
raw_face_nodes_i = np.where(np.isfinite(raw_face_nodes), raw_face_nodes, 0).astype("int64")

tri_list = []
tri_parent_face = []

for i, face in enumerate(raw_face_nodes_i):
    # UGRID connectivity is typically 1-based with 0/-1 as fill values.
    valid = face[face > 0] - 1
    n_valid = valid.size

    if n_valid < 3:
        continue
    if n_valid == 3:
        tri_list.append(valid)
        tri_parent_face.append(i)
        continue

    if n_valid == 4:
        a, b, c, d = valid
        diag_ac = (node_x[a] - node_x[c]) ** 2 + (node_y[a] - node_y[c]) ** 2
        diag_bd = (node_x[b] - node_x[d]) ** 2 + (node_y[b] - node_y[d]) ** 2

        if diag_ac <= diag_bd:
            tri_list.extend([[a, b, c], [a, c, d]])
        else:
            tri_list.extend([[a, b, d], [b, c, d]])
        tri_parent_face.extend([i, i])
        continue

    # Fallback for polygons with >4 nodes: simple fan triangulation.
    for j in range(1, n_valid - 1):
        tri_list.append([valid[0], valid[j], valid[j + 1]])
        tri_parent_face.append(i)

tri_faces = np.asarray(tri_list, dtype="int64")
tri_parent_face = np.asarray(tri_parent_face, dtype="int64")

#%% Make the dataset Parcels-combatible
zf = [0, 1]
zc = [0.5]

# Parcels requires a purely triangular UGRID mesh.
uxgrid = ux.Grid.from_topology(
    node_lon=ds["Mesh_node_x"],
    node_lat=ds["Mesh_node_y"],
    face_node_connectivity=tri_faces,
    fill_value=-1,
)

# Fast remap from original faces -> triangular faces using NumPy indexing.
tri_parent_idx = tri_parent_face.astype(np.intp, copy=False)

def to_face_3d(var_name, target_name):
    src = ds[var_name].transpose("time", "nMesh_face").astype("float32")
    tri_vals = np.take(src.values, tri_parent_idx, axis=1)
    return xr.DataArray(
        tri_vals[:, np.newaxis, :],
        dims=("time", "zc", "n_face"),
        coords={"time": ds["time"], "zc": zc},
        name=target_name,
        attrs=src.attrs,
    )

velocity_data = {
    "U": to_face_3d("velu", "U"),
    "V": to_face_3d("velv", "V"),
}

uxds = ux.UxDataset(
    xr.Dataset(
        velocity_data,
        coords={"zf": ("zf", zf), "zc": ("zc", zc), "time": ds["time"]},
    ),
    uxgrid=uxgrid,
)
fieldset = parcels.FieldSet.from_ugrid_conventions(uxds, mesh="spherical")

#%% Create the simulation
release = 'coast'  # 'coast' or 'off_shore'

if release=='off_shore':
    lat0 = 52.10
    lon0 = 3.52
    day0=5
elif release=='coast':
    lat0 = 52.020000
    lon0 = 4.097500
    day0=19

radii = [100, 200, 400, 800, 1600]  # m
n_points = 16
R = 6371000  # Earth radius (m)

lat = [lat0]
lon = [lon0]
for r in radii:
    theta = np.linspace(0, 2*np.pi, n_points, endpoint=False)
    dlat = np.rad2deg((r / R) * np.cos(theta))
    dlon = np.rad2deg((r / (R * np.cos(np.deg2rad(lat0)))) * np.sin(theta))
    lat.extend(lat0 + dlat)
    lon.extend(lon0 + dlon)

release_dt = np.timedelta64(1, "h")
nrepeat = 12
npart = len(lon)
lon = np.broadcast_to(lon, (nrepeat, npart))
lat = np.broadcast_to(lat, (nrepeat, npart))
time_i = np.datetime64("2025-10-01T00:00:00")
time = (
    np.broadcast_to(time_i, (nrepeat, npart))
    + np.arange(0, nrepeat)[:, np.newaxis] * release_dt
)
pset = parcels.ParticleSet(fieldset, x=lon, y=lat, time=time)

output_file = parcels.ParticleFile(
    "output-matroos.parquet",
    outputdt=np.timedelta64(30, "m"),
    mode="w",
)

def DeleteAnyError(particles, fieldset):
    any_error = particles.state >= 50  # This captures all Errors
    particles[any_error].state = parcels.StatusCode.Delete

pset.execute(
    [parcels.kernels.AdvectionRK2, DeleteAnyError],
    runtime=np.timedelta64(14, "D"),
    dt=np.timedelta64(10, "m"),
    output_file=output_file,
)