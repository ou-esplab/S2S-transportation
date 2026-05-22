import pandas as pd
import xarray as xr
import numpy as np
import string
import math
from scipy import stats
from scipy.stats import t,skew, kurtosis
from scipy.stats import linregress, ks_2samp
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity


import geopandas as gpd

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import matplotlib.patches as mpatches
from matplotlib.path import Path
import matplotlib.patheffects as path_effects
from matplotlib.patches import PathPatch
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.io import shapereader


def plot_crashes(df,save_path='../figs/Figure2.png'):
    # --- Style settings ---
    color_pos = '#D55E00'  # orange
    color_neg = '#0072B2'  # blue
    panel_bg = '#f7f7f7'
    font_size = 12

    # --- Prepare data ---
    df = df.copy()
    df['DATE'] = pd.to_datetime(df['DATE'])
    dates = df['DATE']

    values_main = [
        df['FATAL_CRASH_ANOM'].values,
        df['FATAL_CRASH_ANOM_DETREND'].values
    ]
    values_ref = [
        df['FATAL_CRASH_ANOM_TREND'].values,
        df['FATAL_CRASH_ANOM_DETREND'].values
    ]
    labels = ['TREND', '']
    titles = [
        'a) Monthly Fatal Crash Anomalies',
        'b) Detrended Monthly Fatal Crash Anomalies'
    ]

    # --- Figure setup ---
    fig, axs = plt.subplots(nrows=2, figsize=(8.5, 11), facecolor='white')
    fig.patch.set_facecolor(panel_bg)

    # --- Loop over subplots ---
    for i, (ax, v1, v2, label, title) in enumerate(zip(axs, values_main, values_ref, labels, titles)):
        # Bar plot for positive/negative anomalies
        mask_pos = v1 >= 0
        mask_neg = v1 < 0

        ax.bar(dates[mask_pos], v1[mask_pos], width=25, color=color_pos)
        ax.bar(dates[mask_neg], v1[mask_neg], width=25, color=color_neg)

        # Overlay reference line (climatology/trend)
        if i == 0:
            ax.plot(dates, v2, color='black', linewidth=1, label=label)

        # Titles and labels
        ax.set_title(title, fontsize=font_size, fontweight='bold', pad=12)
        ax.set_ylabel('Fatal Precipitation-related Crashes', fontsize=font_size)
        ax.set_xlabel('Date', fontsize=font_size)

        # X-axis: label every 2 years
        ax.xaxis.set_major_locator(mdates.YearLocator(2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

        # Apply tick label formatting
        ax.tick_params(axis='x', labelrotation=45, labelsize=font_size-3)
        ax.tick_params(axis='y', labelsize=font_size, width=1.0, length=3)

        # Horizontal zero line
        ax.axhline(0, color='gray', linewidth=1, linestyle='--')

        # Style
        for spine in ax.spines.values():
            spine.set_linewidth(1.2)
        ax.grid(axis='y', linestyle=':', alpha=0.4)

        # Force tick label font/rotation (overrides date formatter defaults)
        for lbl in ax.get_xticklabels():
            lbl.set_fontsize(font_size - 3)
            lbl.set_rotation(45)
            lbl.set_ha('right')

    # --- Layout ---
    plt.tight_layout(pad=2.0)
    plt.savefig(save_path,dpi=300, bbox_inches='tight')
    plt.show()

def plot_crashes_clim(df, months_list, outname="../figs/Figure2.png"):

    # ---------------- Style ----------------
    color_pos = '#D55E00'   # orange
    color_neg = '#0072B2'   # blue
    panel_bg = '#f7f7f7'
    font_size = 12

    # ---------------- Data prep ----------------
    df = df.copy()
    df['DATE'] = pd.to_datetime(df['DATE'])
    df['MONTH'] = df['DATE'].dt.month

    dates = df['DATE']
    clim = df['CLIM_CRASH'].values
    total = df['FATAL_CRASH_COUNT'].values

    # ---------------- Figure layout ----------------
    fig = plt.figure(figsize=(8.5, 11), facecolor=panel_bg)
    gs = GridSpec(
        nrows=2,
        ncols=1,
        height_ratios=[1, 1.5],   # top smaller, bottom compact
        #hspace=0.35
        hspace=0.5
    )

    ax_ts = fig.add_subplot(gs[0])
    ax_mon = fig.add_subplot(gs[1])

    # =================================================
    # Top panel: time series
    # =================================================
    mask_pos = clim >= 0
    mask_neg = clim < 0

    ax_ts.bar(dates[mask_pos], clim[mask_pos], width=25, color=color_pos)
    ax_ts.bar(dates[mask_neg], clim[mask_neg], width=25, color=color_neg)

    ax_ts.plot(dates, clim, color='black', linewidth=1)

    ax_ts.set_title(
        'a) Monthly Fatal Crash Totals',
        fontsize=font_size,
        fontweight='bold',
        pad=10
    )
    ax_ts.set_ylabel('Fatal Precipitation-related Crashes', fontsize=font_size)

    ax_ts.xaxis.set_major_locator(mdates.YearLocator(2))
    ax_ts.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax_ts.tick_params(axis='x', rotation=45, labelsize=font_size-3)
    ax_ts.tick_params(axis='y', labelsize=font_size)

    ax_ts.axhline(0, color='gray', linestyle='--', linewidth=1)
    ax_ts.grid(axis='y', linestyle=':', alpha=0.4)

    for spine in ax_ts.spines.values():
        spine.set_linewidth(1.2)




    
    # =================================================
    # Bottom panel: annual cycle
    # =================================================
    df_mon = df.sort_values('MONTH')

    ax_mon.bar(
        df_mon['MONTH'],
        df_mon['CLIM_CRASH'],
        color='skyblue',
        zorder=2
    )

    ax_mon.set_xticks(range(1, 13))
    ax_mon.set_xticklabels(
        ['Jan','Feb','Mar','Apr','May','Jun',
         'Jul','Aug','Sep','Oct','Nov','Dec']
    )
    ax_mon.set_ylabel('Climatological Crash Count', fontsize=font_size)
    ax_mon.set_title(
        'b) Average # of Fatal Crashes by Month',
        fontsize=font_size,
        fontweight='bold'
    )
    ax_mon.grid(axis='y', linestyle='--', alpha=0.4)

    # ---------------- Save ----------------
    plt.savefig(outname, dpi=300, bbox_inches='tight')
    plt.show()


def plot_enso_anomaly_maps(el_nino_avgs, la_nina_avgs, state_centers, save_path='../figs/enso_precip_anomalies.png'):

    fig, axes = plt.subplots(nrows=2, figsize=(8.5, 11), subplot_kw={'projection': ccrs.PlateCarree()})

    cmap = plt.cm.BrBG
    vmin, vmax = -0.5, 0.5
    norm = plt.Normalize(vmin=vmin, vmax=vmax)

    titles = ['a) El Niño', 'b) La Niña']
    data_dicts = [el_nino_avgs, la_nina_avgs]

    shpfilename = shapereader.natural_earth(resolution='50m',
                                            category='cultural',
                                            name='admin_1_states_provinces_lakes')
    reader = shapereader.Reader(shpfilename)
    states = list(reader.records())


    for ax, title, data in zip(axes, titles, data_dicts):
        ax.set_extent([-125, -66, 24, 50], crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.LAND, facecolor='#f0f0f0')
        ax.add_feature(cfeature.OCEAN, facecolor='#d0e0ff')
        ax.add_feature(cfeature.STATES, linewidth=1.2, edgecolor='gray')

        patches = []
        values = []

        for state in states:
            abbrev = state.attributes['postal']
            if abbrev in data:
                geom = state.geometry
                if geom.geom_type == 'Polygon':
                    polygons = [geom]
                elif geom.geom_type == 'MultiPolygon':
                    polygons = list(geom.geoms)
                else:
                    continue
                for polygon in polygons:
                    patches.append(Polygon(np.array(polygon.exterior.coords.xy).T))
                    values.append(data[abbrev])

        pc = PatchCollection(patches, cmap=cmap, norm=norm, edgecolor='gray', linewidth=1.2)
        pc.set_array(np.array(values))
        ax.add_collection(pc)

        for state, value in data.items():
            if state in state_centers:

                lon, lat = state_centers[state]
                ax.text(lon, lat, f"{state}\n{value:.2f}", ha='center', va='center',
                        fontsize=8, bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.2'))

        ax.set_title(title, fontsize=14, fontweight='bold', pad=12)

    # Shared colorbar at bottom
    #sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    #sm.set_array([])
    #cbar = fig.colorbar(sm, ax=axes.ravel().tolist(), orientation='horizontal',
    #                    fraction=0.04, pad=0.05)
    #cbar.set_label('Precipitation Anomaly (mm/day)', fontsize=14)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(
        sm,
        ax=axes.ravel().tolist(),
        orientation='horizontal',
        fraction=0.04,
        pad=0.07,
        extend='both'  # <-- enables arrows at both ends
    )
    cbar.set_label('Precipitation Anomaly (mm/day)', fontsize=14)
    #plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.subplots_adjust(left=0.07, right=0.97, top=0.97, bottom=0.2, hspace=0.08)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

    
def plot_cluster_composites(cluster_comp, cmap='RdBu_r', lon_0=260,
                             suptitle="ERA5 Z500 Anomalies DJF 1981–2019", 
                             nrows=2, ncols=2, levels=np.arange(-90, 100, 10),
                             save_path=None, dpi=300):

    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(8.5, 11.0),
                            subplot_kw={'projection': ccrs.NorthPolarStereo(central_longitude=lon_0)})

    axs = axs.flatten()
    titles = cluster_comp['names'].values
    freq=cluster_comp['freq'].values
    k = 0

    # Create circular Path in Axes coordinates
    theta = np.linspace(0, 2 * np.pi, 100)
    circle_x = 0.5 + 0.5 * np.cos(theta)
    circle_y = 0.5 + 0.5 * np.sin(theta)
    circle_path = Path(np.column_stack([circle_x, circle_y]))

    for ax, label in zip(axs, string.ascii_lowercase):
        # Apply circular boundary using Path
        ax.set_boundary(circle_path, transform=ax.transAxes)

        ax.set_extent([-180, 180, 10, 90], crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax.add_feature(cfeature.BORDERS, linewidth=0.3)
        #ax.gridlines(draw_labels=False)

        cs = ax.contourf(cluster_comp['lon'].values,
                         cluster_comp['lat'].values,
                         cluster_comp['z'][k, :, :].values,
                         levels=levels,
                         cmap=cmap,
                         extend='both',
                         transform=ccrs.PlateCarree())

        title_text = f"({label}) {titles[k]} ({freq[k]}%)" if freq is not None else titles[k]
        ax.set_title(title_text, loc='center', fontsize='medium', fontweight='bold')
#        ax.text(0.01, 0.99, f"({label})", transform=ax.transAxes,
#                ha='left', va='top', fontsize='large', fontweight='bold')
        k += 1

    # fig.subplots_adjust(bottom=0.1, top=0.92, hspace=0.25)
    fig.subplots_adjust(bottom=0.25, top=0.92, hspace=0.05)
    cbar_ax = fig.add_axes([0.25, 0.2, 0.5, 0.02])
    cbar = fig.colorbar(cs, cax=cbar_ax, orientation='horizontal')
    cbar.set_label("m")

    fig.suptitle(suptitle, fontsize='x-large')

    if save_path is not None:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')

    plt.show()
    
def plot_wxregimes_state_crash_anomalies(df, state_centers):
    """
    Plot mean daily precipitation anomalies by weather regime (cluster_name) for each U.S. state.

    Parameters
    ----------
    daily_summary : pd.DataFrame
        Must contain columns ['STATE_ABBR', 'cluster_name', 'PRECIP_ANOM_MEAN'].
    state_centers : dict
        Dictionary mapping STATE_ABBR to (lon, lat) for label placement.

    Notes
    -----
    Produces one panel per cluster_name showing mean PRECIP_ANOM_MEAN per state.
    """
    if state_centers is None:
        raise ValueError("state_centers dictionary must be provided")

    # --- Load continental U.S. states ---
    shpfilename = shapereader.natural_earth('50m', 'cultural', 'admin_1_states_provinces_lakes')
    gdf_states = gpd.read_file(shpfilename)
    continental = gdf_states[~gdf_states['postal'].isin(['AK', 'HI', 'PR'])].copy()
    continental = continental.rename(columns={'postal': 'STATE_ABBR'})

    # --- Aggregate mean precipitation anomaly by state and cluster ---
    df_state_summary = (
        df.groupby(['STATE_ABBR', 'CLUSTER_NAME'], as_index=False)
               .agg({'FATAL_CRASH_ANOM_DETREND': 'mean'}))
    

    # --- Merge with geometry ---
    gdf_plot = continental.merge(df_state_summary, on='STATE_ABBR', how='left')

    # --- Setup plotting ---
    regimes = sorted(df['CLUSTER_NAME'].unique())
    
    ncols = 1
    nrows = len(regimes)
    fig, axes = plt.subplots(nrows, 1, figsize=(8.5, 11),
                             subplot_kw={'projection': ccrs.PlateCarree()})
    axes = axes.flatten()

    cmap = 'BrBG'
    abs_max = max(abs(gdf_plot['FATAL_CRASH_ANOM_DETREND'].min()), gdf_plot['FATAL_CRASH_ANOM_DETREND'].max())
    norm = mcolors.TwoSlopeNorm(vmin=-abs_max, vcenter=0, vmax=abs_max)

    # --- Plot each cluster ---
    for i,(ax, regime) in enumerate(zip(axes, regimes)):
        ax.set_extent([-125, -66.5, 24, 50], ccrs.PlateCarree())
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax.add_feature(cfeature.BORDERS, linewidth=0.3)
        ax.add_feature(cfeature.STATES, edgecolor='gray', linewidth=0.3)

        subset = gdf_plot[gdf_plot['CLUSTER_NAME'] == regime]
        subset.plot(column='FATAL_CRASH_ANOM_DETREND', cmap=cmap, norm=norm,
                    linewidth=0.5, edgecolor='black', ax=ax, transform=ccrs.PlateCarree())

        # Add text labels
        for _, row in subset.iterrows():
            x, y = state_centers.get(row['STATE_ABBR'], (None, None))
            if x is not None and y is not None and not pd.isna(row['FATAL_CRASH_ANOM_DETREND']):
                #ax.text(x, y, f"{row['PRECIP_ANOM']:.2f}", ha='center', va='center',
                #        fontsize=8, fontweight='bold')
                ax.text(x,y, f"{row['FATAL_CRASH_ANOM_DETREND']:.2f}", ha='center', va='center',
                        fontsize=8, bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.2'))

        letter = chr(97 + i)  # 97 is 'a'
        ax.set_title(f"({letter}) {regime}", fontsize=13, fontweight='bold')
        
    # --- Remove extra empty panels ---
    for ax in axes[len(regimes):]:
        ax.axis('off')
        
    # --- Shared colorbar ---
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm._A = []
    # move colorbar higher
    cbar_ax = fig.add_axes([0.25, 0.10, 0.5, 0.02])
    fig.subplots_adjust(bottom=0.08)    
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
    cbar.set_label("Mean Crash Anomaly (crashes/day)", fontweight='bold')

    plt.show()
    
def draw_map(ax, r2_df, col, title,state_centers):
    ax.set_extent([-125, -66, 24, 50])
    ax.add_feature(cfeature.LAND, facecolor='#f0f0f0')
    ax.add_feature(cfeature.OCEAN, facecolor='#d0e0ff')
    ax.add_feature(cfeature.STATES, edgecolor='gray', linewidth=1)

    patches_sig, values_sig = [], []   # significant → colormap
    patches_nonsig = []                # non-significant → gray

    shp = shapereader.natural_earth('50m', 'cultural', 'admin_1_states_provinces')
    for state in shapereader.Reader(shp).records():
        abbr = state.attributes['postal']
        row = r2_df[r2_df['STATE_ABBR'] == abbr]
        if row.empty:
            continue

        r2_val = row.iloc[0][col]
        sig    = row.iloc[0]['significant']
        geom   = state.geometry
        polys  = [geom] if geom.geom_type == 'Polygon' else geom.geoms

        for poly in polys:
            coords = list(poly.exterior.coords)
            poly_patch = Polygon(coords, closed=True)

            if sig:
                patches_sig.append(poly_patch)
                values_sig.append(r2_val)
            else:
                poly_patch.set_facecolor('#d3d3d3')   # <-- gray fill for non-sig
                poly_patch.set_edgecolor('gray')
                patches_nonsig.append(poly_patch)

    # ---- Add non-significant states first (gray) ----
    for p in patches_nonsig:
        ax.add_patch(p)

    # ---- Now add significant states with colormap ----
    cmap = "magma"
    norm = mcolors.Normalize(vmin=0, vmax=0.5)

    pc = PatchCollection(
        patches_sig, edgecolor='gray',
        linewidth=1, cmap=cmap, norm=norm
    )
    pc.set_array(np.array(values_sig))
    ax.add_collection(pc)

    # ---- Labels ----
    for _, row in r2_df.iterrows():
        abbr = row['STATE_ABBR']
        if abbr in state_centers:
            lon, lat = state_centers[abbr]
            ax.text(lon, lat, f"{abbr}\n{row[col]:.2f}",
                    ha='center', va='center', fontsize=8,
                    bbox=dict(facecolor='white', alpha=0.7,
                              boxstyle='round,pad=0.2'))

    ax.set_title(title, fontsize=13, weight='bold')
    return pc


def plot_r2_map_panels(r2_df_list, state_centers, save_path=""):


    # ---------- Two-panel FIGURE ----------
    fig, axs = plt.subplots(
        2, 1, figsize=(8.5, 11),
        subplot_kw={'projection': ccrs.PlateCarree()}
    )

    # panels a and b correspond to list elements 0 and 1
    titles = [
        "a) R² between Total Precip Anomaly and Fatal Crash Anomaly",
        "b) R² between ENSO Precip Anomaly and Fatal Crash Anomaly"
    ]

    pcs = []
    for ax, r2_df, title in zip(axs, r2_df_list, titles):
        pc = draw_map(ax, r2_df, "R2", title, state_centers)
        pcs.append(pc)

    # Shared colorbar
    cbar = fig.colorbar(
        pcs[-1], ax=axs,
        orientation='horizontal', fraction=0.035, pad=0.03, extend='max'
    )
    cbar.set_label("R² Value", fontsize=13)

    plt.tight_layout()
    plt.subplots_adjust(top=0.93, bottom=0.15)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    

    
def plot_monthly_and_djf_map(
    df_monthly_national,
    df_monthly_state,
    df_vehicle_miles,
    months_list,
    state_centers,
    save_path=None,
    figsize=(12, 14)
):
    """
    Plot monthly climatology bar chart and DJF state map normalized by VMT.

    Parameters
    ----------
    df_monthly_national : pd.DataFrame
        Must contain columns ['DATE', 'CLIM_CRASH'].
    df_monthly_state : pd.DataFrame
        Must contain columns ['DATE', 'STATE_ABBR', 'CLIM_CRASH'].
    df_vehicle_miles : pd.DataFrame
        Must contain columns ['STATE_ABBR', 'Total_VMT'].
    state_centers : dict
        Dictionary mapping STATE_ABBR to (lon, lat) for annotation.
    save_path : str, optional
        Path to save the figure. If None, figure is not saved.
    figsize : tuple
        Figure size.
    """
    # --- Preprocess monthly national data ---
    df_monthly_national = df_monthly_national.copy()
    df_monthly_national['DATE'] = pd.to_datetime(df_monthly_national['DATE'])
    df_monthly_national['MONTH'] = df_monthly_national['DATE'].dt.month
    df_plot = df_monthly_national.sort_values('MONTH')

    # Identify DJF months
    df_plot['DJF'] = df_plot['MONTH'].isin(months_list)

    # --- Preprocess monthly state data ---
    df_monthly_state = df_monthly_state.copy()
    df_monthly_state['DATE'] = pd.to_datetime(df_monthly_state['DATE'])
    df_monthly_state['MONTH'] = df_monthly_state['DATE'].dt.month
    df_djf = df_monthly_state[df_monthly_state['MONTH'].isin([12, 1, 2])]

    # Compute DJF sum per state
    djf_avg_by_state = (
        df_djf.groupby('STATE_ABBR', as_index=False)['CLIM_CRASH']
        .sum()
        .rename(columns={'CLIM_CRASH': 'CLIM_CRASH_DJF_SUM'})
    )

    # --- Create figure ---
    fig = plt.figure(figsize=figsize)

    # --- Top panel: bar chart ---
    ax1 = fig.add_subplot(2, 1, 1)
    ax1.bar(
        df_plot['MONTH'],
        df_plot['CLIM_CRASH'],
        color='skyblue',
        edgecolor='none',
        zorder=2
    )
    ax1.set_xticks(range(1,13))
    ax1.set_xticklabels(['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])
    ax1.set_ylabel('Climatological Crash Count')
    ax1.set_title('a) Average # of Fatal Crashes by Month', fontsize=14, fontweight='bold')
    ax1.grid(axis='y', linestyle='--', alpha=0.4)

    # --- Bottom panel: DJF state map ---
    ax2 = fig.add_subplot(2, 1, 2, projection=ccrs.LambertConformal())
    ax2.set_extent([-125, -66, 24, 50], crs=ccrs.PlateCarree())
    ax2.add_feature(cfeature.LAND, facecolor='#f0f0f0')
    ax2.add_feature(cfeature.OCEAN, facecolor='#d0e0ff')
    ax2.add_feature(cfeature.BORDERS, linewidth=1.2)
    ax2.add_feature(cfeature.STATES, linewidth=1.2, edgecolor='gray')

    # Load US states shapefile
    shpfilename = shapereader.natural_earth(resolution='50m',
                                            category='cultural',
                                            name='admin_1_states_provinces_lakes')
    reader = shapereader.Reader(shpfilename)
    states = list(reader.records())

    # Define colormap and normalization
    values = djf_avg_by_state.merge(df_vehicle_miles, on='STATE_ABBR')
    values['norm_crash'] = values['CLIM_CRASH_DJF_SUM'] / values['Total_VMT']
    cmap = plt.cm.Reds
    norm = mcolors.Normalize(vmin=values['norm_crash'].min(), vmax=values['norm_crash'].max())

    # Add colored state polygons
    for state in states:
        abbrev = state.attributes['postal']
        if abbrev in values['STATE_ABBR'].values:
            geom = state.geometry
            value = values.loc[values['STATE_ABBR']==abbrev, 'norm_crash'].iloc[0]
            ax2.add_geometries([geom], crs=ccrs.PlateCarree(),
                               facecolor=cmap(norm(value)),
                               edgecolor='gray', linewidth=1.2)

    # Annotate state values at centroids
    for _, row in values.iterrows():
        state = row['STATE_ABBR']
        value = row['norm_crash']
        if state in state_centers:
            lon, lat = state_centers[state]
            ax2.text(lon, lat, f"{state}\n{value:.1f}", ha='center', va='center',
                     fontsize=8, bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.2'),
                     transform=ccrs.PlateCarree())

    ax2.set_title('b) Average Total # of Fatal Crashes by State (DJF) Normalized by Vehicle Miles Driven',
                  fontsize=14, fontweight='bold')

    # --- Colorbar below map only ---
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax2, orientation='horizontal', fraction=0.04, pad=0.08, extend='both')
    cbar.set_label('Normalized Average # Crashes during DJF', fontsize=12)

    # Adjust layout
    plt.subplots_adjust(left=0.05, right=0.95, top=0.97, bottom=0.1, hspace=0.15)

    # Save figure if path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_wxregimes_state_precip(df, state_centers, save_path='../figs/Figure6.png'):
    """
    Plot mean daily precipitation anomalies by weather regime for U.S. states,
    optimized for landscape publication (11"x8.5") with proper aspect, 
    readable state labels, and automatic handling of overlapping labels.
    """

    if state_centers is None:
        raise ValueError("state_centers dictionary must be provided")

    # --- Load continental U.S. states ---
    shpfilename = shapereader.natural_earth('50m', 'cultural', 'admin_1_states_provinces_lakes')
    gdf_states = gpd.read_file(shpfilename)
    continental = gdf_states[~gdf_states['postal'].isin(['AK', 'HI', 'PR'])].copy()
    continental = continental.rename(columns={'postal': 'STATE_ABBR'})

    # --- Aggregate mean precipitation anomaly by state and cluster ---
    df_state_summary = df.groupby(['STATE_ABBR', 'CLUSTER_NAME'], as_index=False).agg({'PRECIP_ANOM': 'mean'})

    # --- Merge with geometry ---
    gdf_plot = continental.merge(df_state_summary, on='STATE_ABBR', how='left')

    # --- Setup plotting ---
    regimes = sorted(df['CLUSTER_NAME'].dropna().unique())
    ncols = 2
    nrows = math.ceil(len(regimes) / ncols)
    fig = plt.figure(figsize=(11, 8.5))
    gs = GridSpec(nrows, ncols, figure=fig, wspace=0.05, hspace=0.18)

    cmap = 'BrBG'
    abs_max = max(abs(gdf_plot['PRECIP_ANOM'].min()), gdf_plot['PRECIP_ANOM'].max())
    norm = mcolors.TwoSlopeNorm(vmin=-abs_max, vcenter=0, vmax=abs_max)

    # --- Offsets for crowded states ---
    state_label_offsets = {
        'NJ': (0.5, 0), 'DE': (0.5, -0.2), 'MD': (0.5, 0.2),
        'DC': (0, 0.3), 'CT': (0.2, 0.1), 'RI': (0.3, 0.1)
    }
    small_states = state_label_offsets.keys()
    label_fontsize = 6  # general font size for multi-panel

    # --- Plot each cluster ---
    for i, regime in enumerate(regimes):
        row = i // ncols
        col = i % ncols
        ax = fig.add_subplot(gs[row, col], projection=ccrs.LambertConformal())
        ax.set_extent([-125, -66.5, 24, 50], crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax.add_feature(cfeature.BORDERS, linewidth=0.3)
        ax.add_feature(cfeature.STATES, edgecolor='gray', linewidth=0.3)

        subset = gdf_plot[gdf_plot['CLUSTER_NAME'] == regime]
        subset.plot(column='PRECIP_ANOM', cmap=cmap, norm=norm,
                    linewidth=0.5, edgecolor='black', ax=ax, transform=ccrs.PlateCarree())

        # Add state labels
        #for _, row_data in subset.iterrows():
        #    x, y = state_centers.get(row_data['STATE_ABBR'], (None, None))
        #    if x is not None and y is not None and not np.isnan(row_data['PRECIP_ANOM']):
        #        dx, dy = state_label_offsets.get(row_data['STATE_ABBR'], (0, 0))
        #        fontsize = label_fontsize - 1 if row_data['STATE_ABBR'] in small_states else label_fontsize
        #        ax.text(x + dx, y + dy, f"{row_data['PRECIP_ANOM']:.2f}",
        #                ha='center', va='center',
        #                fontsize=fontsize,
        #                bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.2'))

        letter = chr(97 + i)
        ax.set_title(f"({letter}) {regime}", fontsize=12, fontweight='bold', pad=8)
        ax.set_xticks([])
        ax.set_yticks([])

    # Remove empty panels
    for j in range(len(regimes), nrows * ncols):
        fig.add_subplot(gs[j // ncols, j % ncols], projection=ccrs.LambertConformal()).axis('off')

    # --- Shared colorbar ---
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm._A = []
    cbar_ax = fig.add_axes([0.22, 0.02, 0.56, 0.03])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
    cbar.set_label("Precip Anomaly (mm/day)", fontweight='bold')

    # Adjust layout for landscape paper
    fig.tight_layout(rect=[0, 0.06, 1, 0.98])
    plt.savefig(save_path,dpi=300, bbox_inches='tight')
    plt.show()
    
    

def plot_crash_anomalies_cv_smooth(df, cluster_col='CLUSTER_NAME', anomaly_col='FATAL_CRASH_ANOM',
                                   output_file='../figs/Figure8_cv_smooth.png', figsize=(12,10),
                                   smooth_factor=1.5):
    """
    Two-panel figure:
    Top = bar plot of mean crash anomaly by weather regime with 95% CI
    Bottom = standardized PDFs of crash anomalies by regime using cross-validated KDE bandwidth,
             with an optional smoothing factor for publication-quality curves.
    Also prints summary statistics: mean, std, skewness, kurtosis, CV bandwidth.
    """
    fig, axs = plt.subplots(2, 1, figsize=figsize)
    ax_top, ax_bottom = axs

    # ---------------------------
    # Regimes and colors
    # ---------------------------
    regimes = sorted(df[cluster_col].dropna().unique())
    
    # High-contrast colors
    base_colors = ['#E41A1C', '#377EB8', '#4DAF4A', '#984EA3']  # red, blue, green, purple
    if len(regimes) > len(base_colors):
        raise ValueError(f"More regimes ({len(regimes)}) than defined colors ({len(base_colors)}).")
    reg_color_dict = {reg: base_colors[i] for i, reg in enumerate(regimes)}

    print("=== Summary statistics per regime ===")
    bw_dict = {}

    # ---------------------------
    # Top panel — Bar Plot
    # ---------------------------
    mean_anomaly = df.groupby(cluster_col)[anomaly_col].mean()
    counts = df.groupby(cluster_col).size()
    effective_n = counts.astype(int)

    ci = {}
    for reg in regimes:
        sample = df.loc[df[cluster_col] == reg, anomaly_col].dropna()
        sem = sample.sem()
        dfree = effective_n[reg] - 1
        ci[reg] = t.ppf(0.975, df=dfree) * sem if dfree > 0 else np.nan

        # Cross-validated bandwidth
        data = sample.values[:, None]
        if len(data) >= 2:
            bw_candidates = np.logspace(-1, np.log10(50), 30)  
            grid = GridSearchCV(KernelDensity(kernel='gaussian'), {'bandwidth': bw_candidates}, cv=5)
            grid.fit(data)
            best_bw = grid.best_params_['bandwidth']
            bw_dict[reg] = best_bw
        else:
            best_bw = np.nan
            bw_dict[reg] = np.nan

        # Compute stats
        mean_val = sample.mean()
        std_val = sample.std()
        skew_val = skew(sample)
        kurt_val = kurtosis(sample)
        print(f"{reg}: n={len(sample)}, mean={mean_val:.3f}, std={std_val:.3f}, "
              f"skew={skew_val:.3f}, kurtosis={kurt_val:.3f}, CV bw={best_bw:.3f}")

    ax_top.bar(
        regimes,
        [mean_anomaly[reg] for reg in regimes],
        yerr=[ci[reg] for reg in regimes],
        color=[reg_color_dict[reg] for reg in regimes],
        edgecolor='black',
        capsize=8,
        linewidth=1.2
    )
    ax_top.axhline(0, color='gray', linestyle='--')
    ax_top.set_ylabel('Mean Crash Anomaly')
    ax_top.set_title('a) CONUS FPRCA by Weather Regime (DJF)')

    # ---------------------------
    # Bottom panel — PDFs
    # ---------------------------
    x_grid = np.linspace(df[anomaly_col].min(), df[anomaly_col].max(), 500)[:, None]

    for reg in regimes:
        sample = df.loc[df[cluster_col] == reg, anomaly_col].dropna()
        data = sample.values[:, None]
        if len(data) < 2:
            continue

        # Apply smoothing factor to CV bandwidth for plotting
        plot_bw = bw_dict[reg] * smooth_factor

        kde = KernelDensity(kernel='gaussian', bandwidth=plot_bw)
        kde.fit(data)

        log_pdf = kde.score_samples(x_grid)
        pdf = np.exp(log_pdf)
        pdf /= pdf.max()

        ax_bottom.plot(x_grid.flatten(), pdf, color=reg_color_dict[reg], linewidth=2.5, label=reg)
        ax_bottom.axvline(sample.mean(), color=reg_color_dict[reg], linestyle='--', alpha=0.9)

    # ---------------------------
    # Overall distribution
    # ---------------------------
    all_data = df[anomaly_col].dropna().values[:, None]
    if len(all_data) >= 2:
        bw_candidates = np.logspace(-1, np.log10(50), 30)  
        grid_all = GridSearchCV(KernelDensity(kernel='gaussian'), {'bandwidth': bw_candidates}, cv=5)
        grid_all.fit(all_data)
        best_bw_all = grid_all.best_params_['bandwidth']

        # Apply smoothing factor for plotting
        plot_bw_all = best_bw_all * smooth_factor

        kde_all = KernelDensity(kernel='gaussian', bandwidth=plot_bw_all)
        kde_all.fit(all_data)

        log_pdf_all = kde_all.score_samples(x_grid)
        pdf_all = np.exp(log_pdf_all)
        pdf_all /= pdf_all.max()

        mean_val = all_data.mean()
        std_val = all_data.std()
        skew_val = skew(all_data.flatten())
        kurt_val = kurtosis(all_data.flatten())
        print(f"All Regimes: n={len(all_data)}, mean={mean_val:.3f}, std={std_val:.3f}, "
              f"skew={skew_val:.3f}, kurtosis={kurt_val:.3f}, CV bw={best_bw_all:.3f}")

        ax_bottom.plot(x_grid.flatten(), pdf_all, color='gray', linewidth=3, label='All Regimes')
        ax_bottom.axvline(all_data.mean(), color='black', linestyle='--', alpha=0.8)

    ax_bottom.set_xlabel('Crash Anomaly')
    ax_bottom.set_ylabel('Standardized Density')
    ax_bottom.set_title('b) Standardized PDFs of Crash Anomalies by Weather Regime (DJF)')
    ax_bottom.legend()
    ax_bottom.grid(True, linestyle=':', alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.show()

def plot_wxregimes_state_crash_anomalies_2x2(df, state_centers=None, save_path='../figs/Figure9.png'):
    """
    Plot mean daily crash anomalies by weather regime for each U.S. state in a 2x2 panel layout.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns ['STATE_ABBR', 'CLUSTER_NAME', 'FATAL_CRASH_ANOM_DETREND'].
    state_centers : dict, optional
        Not used; state labels removed.

    Notes
    -----
    Produces one panel per cluster_name showing mean FATAL_CRASH_ANOM_DETREND per state.
    """

    # --- Load continental U.S. states ---
    shpfilename = shapereader.natural_earth('50m', 'cultural', 'admin_1_states_provinces_lakes')
    gdf_states = gpd.read_file(shpfilename)
    continental = gdf_states[~gdf_states['postal'].isin(['AK', 'HI', 'PR'])].copy()
    continental = continental.rename(columns={'postal': 'STATE_ABBR'})

    # --- Aggregate mean crash anomaly by state and cluster ---
    df_state_summary = df.groupby(['STATE_ABBR', 'CLUSTER_NAME'], as_index=False) \
                         .agg({'FATAL_CRASH_ANOM_DETREND': 'mean'})

    # --- Merge with geometry ---
    gdf_plot = continental.merge(df_state_summary, on='STATE_ABBR', how='left')

    # --- Setup plotting ---
    regimes = sorted(df['CLUSTER_NAME'].unique())
    n_panels = len(regimes)
    nrows, ncols = 2, 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 10),
                             subplot_kw={'projection': ccrs.PlateCarree()})
    axes = axes.flatten()

    cmap = 'RdBu_r'  # Red = more crashes, Blue = fewer crashes
    norm = mcolors.TwoSlopeNorm(vmin=-0.3, vcenter=0, vmax=0.3)
    
    #abs_max = max(abs(gdf_plot['FATAL_CRASH_ANOM_DETREND'].min()), 
                 # gdf_plot['FATAL_CRASH_ANOM_DETREND'].max())
    #norm = mcolors.TwoSlopeNorm(vmin=-abs_max, vcenter=0, vmax=abs_max)

    # --- Plot each cluster ---
    for i, regime in enumerate(regimes):
        ax = axes[i]
        ax.set_extent([-125, -66.5, 24, 50], ccrs.PlateCarree())
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax.add_feature(cfeature.BORDERS, linewidth=0.3)
        ax.add_feature(cfeature.STATES, edgecolor='gray', linewidth=0.3)

        subset = gdf_plot[gdf_plot['CLUSTER_NAME'] == regime]
        subset.plot(column='FATAL_CRASH_ANOM_DETREND', cmap=cmap, norm=norm,
                    linewidth=0.5, edgecolor='black', ax=ax, transform=ccrs.PlateCarree())

        # Add panel letter
        letter = chr(97 + i)  # 'a', 'b', 'c', 'd'
        ax.set_title(f"({letter}) {regime}", fontsize=13, fontweight='bold')

        # Annotate state values at centroids
       # for _, row in subset.iterrows():
       #     state = row['STATE_ABBR']
       #     value = row['FATAL_CRASH_ANOM_DETREND']
       #     if state in state_centers:
       #         lon, lat = state_centers[state]
       #         ax.text(lon, lat, f"{value:.2f}", ha='center', va='center',
       #                 fontsize=8, bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.2'),
       #                 transform=ccrs.PlateCarree())

    # --- Turn off any extra axes ---
    for ax in axes[n_panels:]:
        ax.axis('off')

    # --- Shared colorbar ---
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm._A = []
    cbar_ax = fig.add_axes([0.25, 0.08, 0.5, 0.02])  # horizontal colorbar
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
    cbar.set_label("Mean Crash Anomaly (crashes/day)", fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()


def plot_state_crash_pdfs_cv_all(df, months_list=[12,1,2], states=['TX','CA','WA'],
                                 pdf_colors=['#E41A1C', '#377EB8', '#4DAF4A', '#984EA3'],
                                 smooth_factor=1.5):
    """
    Plot standardized PDFs of crash anomalies by weather regime for specified states and DJF months.
    Uses cross-validated KDE bandwidths, prints stats, smooths PDFs for plotting, and adds an
    'All Regimes' PDF for comparison.
    """

    # Select DJF months
    df_djf = df[df['DATE'].dt.month.isin(months_list)]

    # x-grid for PDFs
    x_grid = np.linspace(df_djf['FATAL_CRASH_ANOM'].min(),
                         df_djf['FATAL_CRASH_ANOM'].max(), 500)[:, None]

    for state in states:
        df_state = df_djf[df_djf['STATE_ABBR'] == state]
        regimes = sorted(df_state['CLUSTER_NAME'].dropna().unique())
        
        plt.figure(figsize=(10, 6))
        data_dict = {}
        bw_dict = {}

        print(f"\n=== {state} Crash Anomalies Summary Statistics ===")

        # --- Per-regime stats and PDFs ---
        for i, reg in enumerate(regimes):
            data = df_state[df_state['CLUSTER_NAME'] == reg]['FATAL_CRASH_ANOM'].dropna()
            if len(data) < 2:
                continue
            data_dict[reg] = data

            # Cross-validated bandwidth
            bw_candidates = np.logspace(-1, np.log10(50), 30)
            print(bw_candidates)
            grid = GridSearchCV(KernelDensity(kernel='gaussian'),
                                {'bandwidth': bw_candidates}, cv=5)
            grid.fit(data.values[:, None])
            bw_cv = grid.best_params_['bandwidth']
            bw_dict[reg] = bw_cv

            # Print summary stats
            print(f"{reg}: n={len(data)}, mean={data.mean():.3f}, std={data.std():.3f}, "
                  f"skew={skew(data):.3f}, kurtosis={kurtosis(data):.3f}, CV_bw={bw_cv:.3f}")

            # Smoothed PDF for plotting
            plot_bw = bw_cv * smooth_factor
            kde = KernelDensity(kernel='gaussian', bandwidth=plot_bw)
            kde.fit(data.values[:, None])
            pdf = np.exp(kde.score_samples(x_grid))
            pdf /= pdf.max()  # standardize

            plt.plot(x_grid.flatten(), pdf,
                     color=pdf_colors[i % len(pdf_colors)],
                     linewidth=2.5, label=reg)
            plt.axvline(data.mean(), color=pdf_colors[i % len(pdf_colors)],
                        linestyle='--', alpha=0.7)

        # --- Overall "All Regimes" ---
        all_data = df_state['FATAL_CRASH_ANOM'].dropna()
        if len(all_data) >= 2:
            # CV bandwidth for all data
            grid_all = GridSearchCV(KernelDensity(kernel='gaussian'),
                                    {'bandwidth': bw_candidates}, cv=5)
            grid_all.fit(all_data.values[:, None])
            bw_all_cv = grid_all.best_params_['bandwidth']

            # Print overall stats
            print(f"All Regimes: n={len(all_data)}, mean={all_data.mean():.3f}, std={all_data.std():.3f}, "
                  f"skew={skew(all_data):.3f}, kurtosis={kurtosis(all_data):.3f}, CV_bw={bw_all_cv:.3f}")

            # Smoothed PDF
            plot_bw_all = bw_all_cv * smooth_factor
            kde_all = KernelDensity(kernel='gaussian', bandwidth=plot_bw_all)
            kde_all.fit(all_data.values[:, None])
            pdf_all = np.exp(kde_all.score_samples(x_grid))
            pdf_all /= pdf_all.max()

            plt.plot(x_grid.flatten(), pdf_all, color='gray', linewidth=3, label='All Regimes')
            plt.axvline(all_data.mean(), color='black', linestyle='--', alpha=0.8)

        # --- KS tests ---
        print(f"\nKolmogorov–Smirnov test results for {state}:")
        pairs = [(r1, r2) for i, r1 in enumerate(regimes) for r2 in regimes[i+1:]]
        for r1, r2 in pairs:
            if r1 in data_dict and r2 in data_dict:
                stat, pval = ks_2samp(data_dict[r1], data_dict[r2])
                print(f"  {r1} vs {r2}: D={stat:.3f}, p={pval:.4f}")

        # --- Plot formatting ---
        plt.title(f'{state} – Crash Anomalies by Weather Regime (DJF)')
        plt.xlabel('Crash Anomaly')
        plt.ylabel('Standardized Density')
        plt.grid(True, linestyle=':', alpha=0.5)
        plt.legend()
        plt.tight_layout()
        plt.show()
        
        
def plot_djf_state_anomaly_maps(df_monthly_state, df_vehicle_miles, months_list, state_centers, save_path=None, figsize=(12,12)):

    # --- Load US states shapefile ---
    shpfilename = shapereader.natural_earth(resolution='50m',
                                            category='cultural',
                                            name='admin_1_states_provinces_lakes')
    reader = shapereader.Reader(shpfilename)
    states = [s for s in reader.records() if s.attributes['admin']=='United States of America']

    # --- Preprocess monthly state data ---
    df_monthly_state = df_monthly_state.copy()
    df_monthly_state['DATE'] = pd.to_datetime(df_monthly_state['DATE'])
    df_monthly_state['MONTH'] = df_monthly_state['DATE'].dt.month
    df_djf = df_monthly_state[df_monthly_state['MONTH'].isin(months_list)]
    
    
    # Compute DJF avg per state
    djf_avg_by_state = (
        df_djf.groupby('STATE_ABBR', as_index=False)['CLIM_CRASH']
        .mean()
        .round(0)
        .rename(columns={'CLIM_CRASH': 'CLIM_CRASH_DJF_SUM'})
    )
    
    values = djf_avg_by_state.merge(df_vehicle_miles, on='STATE_ABBR')
    values['norm_crash'] = (values['CLIM_CRASH_DJF_SUM'] / values['Total_VMT'])

    # Prepare maps and titles
    data_cols = ['CLIM_CRASH_DJF_SUM', 'norm_crash']
    titles = ['a) Average DJF Fatal Crashes',
              'b) Average DJF Normalized Fatal Crashes']
    
    # --- Create figure with map projections ---
    fig, axs = plt.subplots(2, 1, figsize=figsize, subplot_kw={'projection': ccrs.LambertConformal()})

    for ax, col, title in zip(axs, data_cols, titles):
        ax.set_extent([-125, -66, 24, 50], crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.LAND, facecolor='#f0f0f0')
        ax.add_feature(cfeature.OCEAN, facecolor='#d0e0ff')
        ax.add_feature(cfeature.BORDERS, linewidth=1.2)
        ax.add_feature(cfeature.STATES, linewidth=1.2, edgecolor='gray')

        # Colormap and normalization
        vmin = values[col].min()
        vmax = values[col].max()
        cmap = plt.cm.Reds #if col=='CLIM_CRASH_DJF_SUM' else plt.cm.Oranges
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

        # Plot states
        for state in states:
            abbrev = state.attributes['postal']
            if abbrev in values['STATE_ABBR'].values:
                value = values.loc[values['STATE_ABBR']==abbrev, col].iloc[0]
                geom = state.geometry
                ax.add_geometries([geom], crs=ccrs.PlateCarree(),
                                  facecolor=cmap(norm(value)),
                                  edgecolor='gray', linewidth=1.0)

        # Annotate state values at centroids
        for _, row in values.iterrows():
            state = row['STATE_ABBR']
            value = row[col]
            if state in state_centers:
                lon, lat = state_centers[state]
                if (col=='CLIM_CRASH_DJF_SUM'):
                    ax.text(lon, lat, f"{int(value)}", ha='center', va='center',
                        fontsize=8, bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.2'),
                        transform=ccrs.PlateCarree())
                else:
                    ax.text(lon, lat, f"{value:.2f}", ha='center', va='center',
                        fontsize=8, bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.2'),
                        transform=ccrs.PlateCarree())

        ax.set_title(title, fontsize=14, fontweight='bold')

        # Colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, orientation='horizontal', fraction=0.04, pad=0.05)

    plt.subplots_adjust(hspace=0.2, left=0.05, right=0.95, top=0.95, bottom=0.05)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
