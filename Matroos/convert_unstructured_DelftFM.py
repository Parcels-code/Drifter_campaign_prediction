from glob import glob
import os
import warnings

import numpy as np
import xarray as xr
import uxarray as ux
from tqdm.auto import tqdm

warnings.filterwarnings(
    "ignore",
    message=(
        "Consolidated metadata is currently not part in the Zarr format 3 "
        "specification. It may not be supported by other zarr implementations "
        "and may change in the future."
    ),
    category=UserWarning,
    module=r"zarr\.api\.asynchronous",
)

DIR = "/storage/shared/oceanparcels/input_data/MatroosWaddenSea/DCSMv7_harmonie/flow/raw"
files = sorted(glob(f"{DIR}/dcsm_fm100m_harmonie_*.nc"))
print(f"Found {len(files)} files in {DIR}.")

# Build a fixed triangular mesh once from the first file.
mesh_ds = xr.open_dataset(files[0])

#%% Make all faces triangular (for uxarray/parcels compatibility)
node_x = mesh_ds["Mesh_node_x"].values
node_y = mesh_ds["Mesh_node_y"].values
raw_face_nodes = mesh_ds["Mesh_face_nodes"].values

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

#%% Make the dataset Parcels-compatible
zf = [0, 1]
zc = [0.5]

# Parcels requires a purely triangular UGRID mesh.
uxgrid = ux.Grid.from_topology(
    node_lon=mesh_ds["Mesh_node_x"],
    node_lat=mesh_ds["Mesh_node_y"],
    face_node_connectivity=tri_faces,
    fill_value=-1,
)

# Fast remap from original faces -> triangular faces using NumPy indexing.
tri_parent_idx = tri_parent_face.astype(np.intp, copy=False)


def to_face_3d(ds, var_name, target_name):
    src = ds[var_name].transpose("time", "nMesh_face").astype("float32")
    tri_vals = np.take(src.values, tri_parent_idx, axis=1)
    return xr.DataArray(
        tri_vals[:, np.newaxis, :],
        dims=("time", "zc", "n_face"),
        coords={"time": ds["time"], "zc": zc},
        name=target_name,
        attrs=src.attrs,
    )


for i, file in enumerate(
    tqdm(files, desc=f"Converting", unit="file"), start=1
):
    new_name = file.replace("/raw/", "/")
    ds = xr.open_dataset(file)

    data = {
        "U": to_face_3d(ds, "velu", "U"),
        "V": to_face_3d(ds, "velv", "V"),
    }

    # if i == 1:
    data.update(
        {
            "Mesh_node_x": mesh_ds["Mesh_node_x"],
            "Mesh_node_y": mesh_ds["Mesh_node_y"],
            "tri_face_nodes": (("n_face", "n_tri_nodes"), tri_faces),
        }
    )
    uxds = ux.UxDataset(
        xr.Dataset(
            data,
            coords={"zf": ("zf", zf), "zc": ("zc", zc), "time": ds["time"]},
        ),
        uxgrid=uxgrid,
    )
    uxds.to_netcdf(new_name, mode="w")
    # else:
    #     append_ds = xr.Dataset(data, coords={"zc": ("zc", zc), "time": ds["time"]})
    #     # append_ds.drop_encoding().to_zarr(
    #     #     combined_name, mode="a", append_dim="time" #, consolidated=False
    #     # )
    #     append_ds.to_zarr(combined_name, mode="a", append_dim="time")

    ds.close()

