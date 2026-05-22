#!/usr/bin/env python
# coding: utf-8

# In[1]:


import geopandas as gpd
import rioxarray
import xarray as xr
import numpy as np
import pandas as pd
from rasterio import features
import rasterio
import dask.array as da
import os

from dask.distributed import Client
from dask.distributed import LocalCluster
from multiprocessing import freeze_support


# In[3]:


def daily_climo(da,varname,clim_fname=None):
  
    # This function is adapted the code written by Ray Bell for the SubX project; it is for the
    # verification data
        
    # Average daily data
    da_day_clim = da.groupby('time.dayofyear').mean('time')
    
    # Rechunk for time
    da_day_clim = da_day_clim.chunk({'dayofyear': 366})
    
    # Pad the daily climatolgy with nans
    x = np.empty((366, len(da_day_clim.lat), len(da_day_clim.lon)))
    x.fill(np.nan)
    _da = xr.DataArray(x,name=varname, coords=[np.linspace(1, 366, num=366, dtype=np.int64),
                              da_day_clim.lat, da_day_clim.lon],
                              dims = da_day_clim.dims)
    da_day_clim_wnan = da_day_clim.combine_first(_da)

    
    # Period rolling twice to make it triangular smoothing
    # See https://bit.ly/2H3o0Mf
    da_day_clim_smooth = da_day_clim_wnan.copy()
 
    

    for i in range(2):
        # Extand the DataArray to allow rolling to do periodic
        da_day_clim_smooth = xr.concat([da_day_clim_smooth[-15:],
                                        da_day_clim_smooth,
                                        da_day_clim_smooth[:15]],
                                        'dayofyear')
        # Rolling mean
        da_day_clim_smooth = da_day_clim_smooth.rolling(dayofyear=31,
                                                        center=True,
                                                        min_periods=1).mean()
        # Drop the periodic boundaries
        da_day_clim_smooth = da_day_clim_smooth.isel(dayofyear=slice(15, -15))

    
    # Extract the original days
    da_day_clim_smooth = da_day_clim_smooth.sel(dayofyear=da_day_clim.dayofyear)

    da_day_clim_smooth.name=varname
    ds_day_clim_smooth=da_day_clim_smooth.to_dataset()
    
    # Save to file if filename provide and return True, otherwise return the data
    if (clim_fname):
        print("Writing CLIM FILE: ",clim_fname)
        ds_day_clim_smooth.to_netcdf(clim_fname)
    else:
        return ds_day_clim_smooth


# In[4]:


def get_state_mask_da(states_gdf, x, y, state_col='STATEFP', cache_path='../data/state_mask.nc', force_rebuild=False):
    """
    Rasterize state polygons to an xarray.DataArray matching a given grid.

    Parameters:
        states_gdf: GeoDataFrame with state polygons.
        x: 1D array of x coordinates (longitude).
        y: 1D array of y coordinates (latitude).
        state_col: Column name in states_gdf with unique state IDs.
        cache_path: Path to save/load the rasterized mask.
        force_rebuild: If True, rebuild mask even if cache file exists.

    Returns:
        state_mask_da: xarray.DataArray with dimensions (y, x), values = state ID.
    """
    if os.path.exists(cache_path) and not force_rebuild:
        print(f"Loading cached state mask from {cache_path}")
        return xr.load_dataarray(cache_path)

    print("Rasterizing state polygons to grid...")

    # Prepare rasterization transform
    transform = rasterio.transform.from_origin(
        west=x.min() - (x[1] - x[0]) / 2,
        north=y.max() + (y[0] - y[1]) / 2,
        xsize=(x[1] - x[0]),
        ysize=abs(y[1] - y[0])
    )

    shapes = zip(states_gdf.geometry, states_gdf[state_col])
    state_mask_array = features.rasterize(
        shapes=shapes,
        out_shape=(len(y), len(x)),
        transform=transform,
        fill=0,
        dtype='int32'
    )

    # Wrap in DataArray
    state_mask_da = xr.DataArray(state_mask_array, coords={'y': y, 'x': x}, dims=('y', 'x'))
    state_mask_da.name = 'state_mask'
    state_mask_da.attrs['description'] = f'State mask from {state_col}'
    
    # Save for reuse
    state_mask_da.to_netcdf(cache_path)
    print(f"Saved state mask to {cache_path}")

    return state_mask_da


# In[5]:


def rasterize_states(states_gdf, x, y, state_col='STATEFP'):
    transform = rasterio.transform.from_origin(
        west=x.min() - (x[1] - x[0]) / 2,
        north=y.max() + (y[0] - y[1]) / 2,
        xsize=(x[1] - x[0]),
        ysize=abs(y[1] - y[0])
    )

    # Ensure state_col is integer
    shapes = zip(states_gdf.geometry, states_gdf[state_col])

    raster = features.rasterize(
        shapes=shapes,
        out_shape=(len(y), len(x)),
        transform=transform,
        fill=0,
        dtype='int32'
    )
    return raster, transform


# In[6]:


def compute_state_anoms_dask(da, states_gdf, state_col, states_mask_file):
    
    """
    Compute spatial mean and total precip anomaly per state and time using xarray and rasterized mask.
    Fully parallel with Dask.
    """
    
    # Ensure CRS alignment
    da = da.rio.write_crs("EPSG:4326")
    states_gdf = states_gdf.to_crs("EPSG:4326")
    
    states_gdf[state_col] = states_gdf[state_col].astype(int)
    
    x=da['x'].values
    y=da['y'].values
    
    # Get rasterized state_mask from file or creat it if it doesn't exist
    if os.path.exists(states_mask_file):
        state_mask=xr.open_dataset(states_mask_file)
    else:
        state_mask = get_state_mask_da(states_gdf, x=x, y=y,cache_path=states_mask_file)
    
    state_mask_array, _ = rasterize_states(states_gdf, x, y, state_col='STATEFP')
    state_mask_da = xr.DataArray(state_mask_array, coords=[('y', y), ('x', x)])   
    print("STATE MASK LIST: ",np.unique(state_mask_da)[0:2])

    # Initialize list of results
    results = []

    for state_id in np.unique(state_mask_da):
        
        state_str = str(state_id).zfill(2)

        print(state_id,state_str)
        
        if state_id == 0:
            continue  # skip background

        # Mask for this state
        mask = (state_mask_da == state_id)

        # Apply mask to precip anomaly
        masked = da.where(mask)
        print(masked)
        
        # Compute daily mean and sum
        mean_ts = masked.mean(dim=['y', 'x'], skipna=True)
        sum_ts = masked.sum(dim=['y', 'x'], skipna=True)
        
        # Combine into DataFrame
        df = xr.merge([
            mean_ts.rename("ANOM_MEAN"),
            sum_ts.rename("ANOM_SUM")
        ]).to_dataframe().reset_index()
        df = df.drop(columns=['dayofyear', 'spatial_ref']).rename(columns={'time': 'DATE'})
        df['STATE'] = state_str
        results.append(df)
        print(df)

    # Concatenate all states
    df_all = pd.concat(results, ignore_index=True)

    # Convert STATE to integer to match FIPS
    df_all['STATE'] = df_all['STATE'].astype(int)
    states_gdf[state_col] = states_gdf[state_col].astype(int)

    df_all = df_all.merge(
        states_gdf[[state_col, 'STUSPS']], 
        left_on='STATE', 
        right_on=state_col, 
        how='left'
    )

    df_all = df_all.rename(columns={'STUSPS': 'STATE_ABBR'})
    print(df_all)
    

    return df_all


def process_var(fname, varname, cfname, outfile, states_gdf, djf_months, sdate, edate, states_mask_file):
    """Process a variable (precip, SWE) and compute state anomalies."""
    print("READING DATA IN PROCESS_VAR") 
    ds = xr.open_mfdataset(fname, combine='by_coords')
    print(ds)
    
    rename_map = {}
    if 'longitude' in ds.coords:
        rename_map['longitude'] = 'lon'
    if 'latitude' in ds.coords:
        rename_map['latitude'] = 'lat'
    if 'X' in ds.coords:
        rename_map['X'] = 'lon'
    if 'Y' in ds.coords:
        rename_map['Y'] = 'lat'
    if rename_map:  # only rename if there's something to rename
        ds = ds.rename(rename_map)
        
    # If first latitude is smaller than last → going south to north
    dslats=ds['lat'].values
    if dslats[0] < dslats[-1]:
        ds = ds.reindex({'lat': list(reversed(ds['lat']))})

    if not np.issubdtype(ds['time'].dtype, np.datetime64):
        ds = ds.assign_coords(time=pd.to_datetime(ds['time'].values))
    ds = ds.sel(time=slice(sdate, edate))

    print("Climo: ",cfname)
    # Climatology
    if os.path.exists(cfname):
        ds_climo = xr.open_dataset(cfname)
    else:
        ds_climo = daily_climo(ds[varname], varname, clim_fname=cfname)

    # Compute anomalies
    print("Computing Anoms")
    ds_anoms = ds[varname].groupby('time.dayofyear') - ds_climo[varname]
    ds_anoms = ds_anoms.sel(time=ds_anoms['time'].dt.month.isin(djf_months))
    anom_da = ds_anoms.rename({'lon': 'x', 'lat': 'y'})
    anom_da = anom_da.chunk({'time': 1, 'y': -1, 'x': -1})

    # Compute state-level anomalies
    print("Calling compute_state_anoms_dask")
    df_state = compute_state_anoms_dask(anom_da, states_gdf,
                                        state_col='STATEFP',
                                        states_mask_file=states_mask_file)
    print("Writing data to: ", outfile)
    df_state.to_csv(outfile, index=False)
    print(f"Saved {varname} state anomalies to {outfile}")

# In[7]:

def main():
    cluster = LocalCluster()
    cluster

    sdate = "1981-01-01"
    edate = "2023-12-31"
    djf_months = np.arange(1,13,1)

    # Load states shapefile once
    states_gdf = gpd.read_file('../data/state_shape_file/cb_2018_us_state_20m.shp')
    print(f"Loaded {len(states_gdf)} states.")

    # Precip
    process_var(
        fname='/data/esplab/shared/obs/gridded/atm/precip/daily/chirps-v2.0/p25/*',
        varname='precip',
        cfname='../data/precip/chirps_p25_climo.nc',
        outfile='../data/state_precip_chirps.csv',
        states_gdf=states_gdf,
        djf_months=djf_months,
        sdate=sdate,
        edate=edate,
        states_mask_file='../data/state_mask_chirps.nc'
    )

    # SWE
 #   print("RUNNING SWE")
 #   process_var(
        #fname='/data/esplab/aelyoussoufi/snow_water_equivalent.nc',
 #       varname='swe',
 #       cfname='../data/swe/climo_swe.nc',
 #       outfile='../data/state_swe.csv',
 #       states_gdf=states_gdf,
 #       djf_months=djf_months,
 #       sdate=sdate,
 #       edate=edate,
 #       states_mask_file='../data/state_mask_swe.nc'
 #   )
 #   print("SWE DONE")
if __name__ == '__main__':
    freeze_support()
    main()
