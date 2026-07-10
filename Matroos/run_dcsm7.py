import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import os

import matplotlib.tri as mtri
import uxarray as ux

import parcels

#%% Open files
files = "/storage/shared/oceanparcels/input_data/MatroosWaddenSea/DCSMv7_harmony/maps2d_dcsm7_harmonie_combined.zarr"
# TODO also add waves ("swan_dcsm_harmony"?)
ds = parcels.open_raw_zarr(files)

uxgrid = ux.Grid.from_topology(
    node_lon=ds["Mesh_node_x"],
    node_lat=ds["Mesh_node_y"],
    face_node_connectivity=ds["tri_face_nodes"],
    fill_value=-1,
)

uxds = ux.UxDataset(
    xr.Dataset(
        { "U": ds["U"], "V": ds["V"] },
        coords={"zf": ("zf", ds["zf"].data), "zc": ("zc", ds["zc"].data), "time": ds["time"]},
    ),
    uxgrid=uxgrid,
)
fieldset = parcels.FieldSet.from_ugrid_conventions(uxds, mesh="spherical")
fieldset.describe()

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
print(f"Running {nrepeat} releases of {npart} particles each, for a total of {nrepeat*npart} particles.")
pset = parcels.ParticleSet(fieldset, x=lon, y=lat, t=time)

slurm_job_id = os.getenv("SLURM_JOB_ID", "local")
output_name = f"output-matroos-{slurm_job_id}.parquet"

output_file = parcels.ParticleFile(
    output_name,
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
