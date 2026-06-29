#%%
#%%

"""
Author: Jimena Medina Rubio (j.medinarubio@uu.nl)
Date: 29/06/2026
"""

#%%
# import needed packages
print('importing packages')
import numpy as np
import math
import xarray as xr
from parcels import FieldSet, ParticleSet,Variable, JITParticle
from parcels import AdvectionRK4
from parcels.tools.converters import Geographic, GeographicPolar 
from datetime import datetime, timedelta
import copernicusmarine

copernicusmarine.login(username="",password="",)
#%%

def add_stokes(particle, fieldset, time):
    Us = fieldset.Ustokes[time, particle.depth,
                                particle.lat, particle.lon]
    Vs = fieldset.Vstokes[time, particle.depth,
                                particle.lat, particle.lon]
    particle_dlon += Us * particle.dt  
    particle_dlat += Vs * particle.dt  


def add_wind(particle, fieldset, time):
    Uwind = fieldset.Uw[time, particle.depth,
                                particle.lat, particle.lon]
    Vwind = fieldset.Vw[time, particle.depth,
                                particle.lat, particle.lon]
    particle_dlon += Uwind * particle.dt* fieldset.wind_coeff  
    particle_dlat += Vwind * particle.dt* fieldset.wind_coeff  


def set_displacement_v2(particle, fieldset, time):
    """ 
    Kernel to check if particles are close to the coast and if they are
    set the particles up for displacement away from the coast. The actual
    displacement of th eparticles is doen int the kernel "displace"

    Dependencies: 
    fieldset with distance to shore (distance2shore)
    fieldste with displacement field away from shore (dispU, dispV)
    """

    new_lat= particle.lat + particle_dlat
    new_lon= particle.lon + particle_dlon

    particle.d2s = fieldset.distance2shore[
        time, particle.depth, new_lat, new_lon
    ]
    if particle.d2s < 0.5:#units: gridcells #only at the coast. otherwise velocity mask is 0
        dispUab = fieldset.dispU[time, particle.depth, new_lat, new_lon]
        dispVab = fieldset.dispV[time, particle.depth, new_lat, new_lon]
        particle.dU = dispUab
        particle.dV = dispVab

        if math.fabs(particle.dU) + math.fabs(particle.dV)>0.: #truly displaced
            particle.n_displacements+=1

            particle_dlon += particle.dU * particle.dt
            particle_dlat += particle.dV * particle.dt

    else:
        particle.dU = 0.0
        particle.dV = 0.0


def sample_land(particle, fieldset, time):
    particle.landmask= fieldset.landmask[time, particle.depth, particle.lat, particle.lon]

def delete_on_land(particle, fieldset, time):
    if particle.landmask >= 0.5: #in practice ==1 because interp = nearest
        particle.delete()

def too_close_to_edge(particle, fieldset, time):
    """
    Kernel to delete particles too close to the edge to be able to calculate
    the derivative of the fluid at that location using finite differences

    Dependencies:
    - lon_min,lon_max, lat_min, lat_max, boundaries of domain (fieldset
      constants)
    - delta_x, delta_y, stepsize used for calculating derivatives (fieldset
      constants)
    """
    if(math.fabs(particle.lon - fieldset.lon_min) < 2 * fieldset.delta_x):
        particle.delete()
    if(math.fabs(particle.lat - fieldset.lat_min) < 2 * fieldset.delta_y):
        particle.delete()
    if(math.fabs(fieldset.lon_max - particle.lon) < 2 * fieldset.delta_x):
        particle.delete()
    if(math.fabs(fieldset.lat_max - particle.lat) < 2 * fieldset.delta_y):
        particle.delete()


def remove_at_bounds(particle, fieldset, time):
    """
    Kernel for deleting particles if they outside the boundary of the
    simulation domaim. 
    
    Dependencies:
    - lon_min,lon_max, lat_min, lat_max, boundaries of domain 
    """
    flag_ = False
    if particle.lat < fieldset.lat_min:
        particle.delete()
        flag_ = True
    if particle.lat > fieldset.lat_max:
        particle.delete()        
        flag_ = True
    if particle.lon < fieldset.lon_min:
        particle.delete()
        flag_ = True
    if particle.lon > fieldset.lon_max:
        particle.delete()
        flag_ = True
#%%

def run_experiment(year, month, day, hour, minute,days_simulation, wind_coeff, inner_dt, output_dt, dt_release,
                   output_file, landmaskfile, lon_particles, lat_particles):
    

    print(f'Starting simulation: {day}-{month}-{year} at {hour}:{minute}')
    release_date = datetime(year, month, day-1) 
    end_date=  release_date + timedelta(days=days_simulation+2) #including buffer day


    """
    HYDRODYNAMIC MODEL
    """
    variables = {'U': 'uo',
                'V': 'vo'}

    dimensions = {'lat': 'latitude',
                'lon': 'longitude',
                'time': 'time'}
    
    ds=copernicusmarine.open_dataset('cmems_mod_nws_phy-cur_anfc_1.5km-2D_PT15M-i', variables=["uo", "vo"],
                            start_datetime=release_date,
                            end_datetime=end_date)
    fieldset=FieldSet.from_xarray_dataset(ds, variables, dimensions,
                                                                    allow_time_extrapolation=False)
    print('fieldset created')

    """
    WAVE MODEL
    """
    variables_stokes = {'Ustokes': 'VSDX',
                'Vstokes': 'VSDY'}

    dimensions_stokes = {'lat': 'latitude',
                'lon': 'longitude',
                'time': 'time'}

    
    ds_waves = copernicusmarine.open_dataset(
                dataset_id="cmems_mod_nws_wav_anfc_1.5km_PT1H-i",
                variables=["VSDX", "VSDY"],
                start_datetime=release_date,
                end_datetime=end_date)

    fieldset_Stokes= FieldSet.from_xarray_dataset(ds_waves, variables_stokes, dimensions_stokes)
    fieldset_Stokes.Ustokes.units = GeographicPolar()
    fieldset_Stokes.Vstokes.units = Geographic()

    fieldset.add_field(fieldset_Stokes.Ustokes)
    fieldset.add_field(fieldset_Stokes.Vstokes)
    print('Stokes fieldset created')

    """
    WIND
    """

    ds1 = xr.open_dataset("ERA5_wind_2024_2025_october.nc")
    ds2 = xr.open_dataset("ERA5_wind_2024_2025_november.nc")
    ds3 = xr.open_dataset("ERA5_wind_2024_2025_december.nc")

    ds = xr.concat([ds1, ds2, ds3], dim="valid_time")  # or "time"

    fieldset_wind = FieldSet.from_xarray_dataset(
        ds,
        variables={
            "Uw": "u10",
            "Vw": "v10",
        },
        dimensions={
            "lon": "longitude",
            "lat": "latitude",
            "time": "valid_time", 
        },
        mesh="spherical",
    )

    fieldset_wind.Uw.units = GeographicPolar()
    fieldset_wind.Vw.units = Geographic()
    
    fieldset.add_field(fieldset_wind.Uw)
    fieldset.add_field(fieldset_wind.Vw)

    fieldset.add_constant('wind_coeff', wind_coeff)

   
    """
    LANDMASK FIELD
    """
    filenames_landmask = {'landmask':landmaskfile}
    dimensions_landmask = {'lat': 'lat',
                                'lon': 'lon'}

    variables_landmask =  {'landmask':'landmask'}
    fieldset_landmask = FieldSet.from_netcdf(filenames_landmask,
                                                variables_landmask,
                                                dimensions_landmask,
                                                indices={},
                                                mesh='spherical',
                                                allow_time_extrapolation=True)
    fieldset.add_field(fieldset_landmask.landmask)
    fieldset.landmask.interp_method = ('nearest')


    fieldset.add_constant('gradient', True)
    fieldset.add_constant('g', 9.81)
    #radius earth in meters
    fieldset.add_constant('Rearth', 6371 * 10**3)
    # kinematic viscosity water
    fieldset.add_constant('nu',1.3729308666017527*10**(-6))
    fieldset.add_constant('Omega_earth', 7.2921 * (10**-5))

    Delta_x = fieldset.U.grid.lon[1]-fieldset.U.grid.lon[0]
    Delta_y = fieldset.U.grid.lat[1]-fieldset.U.grid.lat[0]

    # stepsize for finite differences calculation
    delta_x = 0.5 * Delta_x
    delta_y = 0.5 * Delta_y
    print(f'delta_x = {delta_x}')

    fieldset.add_constant('delta_x', delta_x)
    fieldset.add_constant('delta_y', delta_y)
    
    lon_min = fieldset.U.grid.lon[1]
    lat_min = fieldset.U.grid.lat[1]
    lon_max = fieldset.U.grid.lon[-2]
    lat_max = fieldset.U.grid.lat[-2]
    print(f'lon domain = {lon_min} - {lon_max}')
    print(f'lat domain = {lat_min} - {lat_max}')

    fieldset.add_constant('lon_min', lon_min)
    fieldset.add_constant('lon_max', lon_max)

    fieldset.add_constant('lat_min', lat_min)
    fieldset.add_constant('lat_max', lat_max)

    #create_variables
    class InertialParticle2D(JITParticle):
        landmask= Variable('landmask', dtype=np.float32, to_write=True, initial=0.)

    inertialparticle = InertialParticle2D

    #KERNELS
    kernels = [too_close_to_edge]
    kernels.append(remove_at_bounds) 
    kernels.append(AdvectionRK4)
    kernels.append(add_stokes) #ADD STOKES DRIFT AFTER RK4 (SEE PLASTIC PARCELS)
    kernels.append(add_wind)
    kernels.append(sample_land)
    kernels.append(delete_on_land)
    

    #RELEASE
    nparticles = lon_particles.size
    starttime = datetime(year, month, day, hour, minute, 0, 0)

    release_interval = timedelta(hours=dt_release)
    n_releases = 7*24   # one release per hour of the week

    lon_all = np.tile(lon_particles, n_releases)
    lat_all = np.tile(lat_particles, n_releases)

    time_all = np.repeat(
        [starttime + i * release_interval for i in range(n_releases)],
        len(lon_particles)
    )

    pset = ParticleSet.from_list(
                fieldset=fieldset,
                pclass=inertialparticle,
                lon=lon_all,
                lat=lat_all,
                time=time_all,
            )
    
    runtime =timedelta(days= days_simulation)
    
    dt_write = timedelta(hours=output_dt)

    pfile = pset.ParticleFile(name=output_file, outputdt=dt_write,)

    dt_timestep = timedelta(minutes=inner_dt)
    pset.execute(kernels, runtime=runtime, dt=dt_timestep, 
                 verbose_progress=True,  output_file=pfile)
    
    print(f"Simulation complete for start release date: {day}-{month}-{year}.")


#%%
#SIMULATION INPUTS
release='coast'
year=2024

month= 11
days_simulation=3
wind_coeff=0.01

inner_dt=5 #minutes
output_dt= 1#hours
release_dt=1 #hours

#%%
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

lat_particles = np.array(lat)
lon_particles = np.array(lon)
#%%
"""
#CODE TO PLOT INITIAL LOCATION

R = 6371000

x = np.deg2rad(lon_particles - lon0) * R * np.cos(np.deg2rad(lat0))

y = np.deg2rad(lat_particles - lat0) * R

plt.figure(figsize=(6, 6))
plt.scatter(x, y, s=25)
plt.scatter(0, 0, s=100, marker="*")
plt.xlabel("x distance from center (m)")
plt.ylabel("y distance from center (m)")
plt.title("Particle release configuration")
plt.axis("equal")
plt.grid(True)
plt.show()
"""

#%%

landmaskfile='NWES_landmask.nc'

output_file= f'output/{release}/{release}_{day0}-{month}-{year}_wind{wind_coeff}_{days_simulation}days.zarr'

run_experiment(year=year,month=month, day=day0, hour=0, minute=0, wind_coeff=wind_coeff,
                days_simulation=days_simulation, inner_dt=inner_dt, output_dt=output_dt, dt_release=release_dt,
                output_file=output_file, landmaskfile=landmaskfile, lon_particles=lon_particles, lat_particles=lat_particles)
#%%
