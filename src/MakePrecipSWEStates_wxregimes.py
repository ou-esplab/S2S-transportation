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

import matplotlib.pyplot as plt
from dask.distributed import Client
from dask.distributed import LocalCluster
from multiprocessing import freeze_support


def get_state_mask_da(states_gdf, x, y, state_col='STATEFP'):
    """
    Rasterize state polygons to an xarray.DataArray matching a given grid.

    Parameters:
        states_gdf: GeoDataFrame with state polygons.
        x: 1D array of x coordinates (longitude).
        y: 1D array of y coordinates (latitude).
        state_col: Column name in states_gdf with unique state IDs.

    Returns:
        state_mask_da: xarray.DataArray with dimensions (y, x), values = state ID.
    """
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


def compute_state_anoms_dask(da, states_gdf, state_col):
    
    """
    Compute spatial mean and total precip anomaly per state and time using xarray and rasterized mask.
    Fully parallel with Dask.
    """
    
    # Ensure CRS alignment
    da = da.rio.write_crs("EPSG:4326")
    states_gdf = states_gdf.to_crs("EPSG:4326")
    
    states_gdf[state_col] = states_gdf[state_col].astype(int)
    print(states_gdf)

    x=da['x'].values
    y=da['y'].values
   
    from shapely.geometry import Point
    
    print("CHecking POINTS before get_state_mask_da")
    pt = Point(x[0], y[0])
    print(any(states_gdf.geometry.contains(pt)))

    state_mask = get_state_mask_da(states_gdf, x=x, y=y)

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
        df = df.drop(columns=['spatial_ref'])
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


def process_var(fname, varname, outfile, states_gdf):
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
    print("Converting longitudes from 0–360 to -180–180")
    ds = ds.assign_coords(lon=((ds['lon'] + 180) % 360 - 180))
    ds = ds.sortby('lon')  # keep in order

    da = ds['precip'].rename({'lon': 'x', 'lat': 'y'})
    da = da.chunk({'cluster':-1,'y': -1, 'x': -1})
    print(da)

    # Compute state-level anomalies
    print("Calling compute_state_anoms_dask")

    df_state = compute_state_anoms_dask(da, states_gdf,
                                        state_col='STATEFP',
                                        )
    print("Writing data to: ", outfile)
    df_state.to_csv(outfile, index=False)
    print(f"Saved {varname} state anomalies to {outfile}")

# In[7]:

def main():
    cluster = LocalCluster()
    cluster

    # Load states shapefile once
    states_gdf = gpd.read_file('../data/state_shape_file/cb_2018_us_state_20m.shp')
    print(f"Loaded {len(states_gdf)} states.")

    # Precip
    process_var(
        fname='../data/wxregimes/era5_cluster_comp_p_na_DJF1981-2019.nc',
        varname='precip',
        outfile='../data/precip/state_precip_wxregimes.csv',
        states_gdf=states_gdf
    )

    # SWE
   # print("RUNNING SWE")
   # process_var(
   #     fname='/data/esplab/aelyoussoufi/snow_water_equivalent.nc',
   #     varname='swe',
   #     cfname='../data/swe/climo_swe.nc',
   #     outfile='../data/state_swe.csv',
   #     states_gdf=states_gdf,
   #     djf_months=djf_months,
   #     sdate=sdate,
   #     edate=edate,
   #     states_mask_file='../data/state_mask_swe.nc'
   # )
   # print("SWE DONE")
if __name__ == '__main__':
    freeze_support()
    main()
