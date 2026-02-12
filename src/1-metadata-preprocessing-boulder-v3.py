# %%
import pandas as pd
import geopandas as gpd
import numpy as np

from geopy.geocoders import Nominatim
import contextily as ctx
import shapely as shp

import os
import glob
import hashlib
import argparse

import matplotlib.pyplot as plt


CFG_ROOT, CFG_EPS = './data', 1e-9
BOULDER_DATASET_META_FEATURES = ['Station_Name', 'Address', 'City', 'State_Province', 'Zip_Postal_Code', 'Port_Type']


# In[9]:
if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='Generate the metadata for the EVCS of the Boulder dataset')
    args = parser.parse_args()


    # Read Boulder Dataset
    df_meta = pd.read_csv(
        os.path.join(CFG_ROOT, 'ev-load-open-data', '1. Input Data', '3. Boulder', 'ev_chargingstationdata_Boulder_March 2021.csv'),
        usecols=BOULDER_DATASET_META_FEATURES,
    ).drop_duplicates(
        keep='last'
    ).rename(
        {
            'Station_Name':'evse_id',
        },
        axis=1,
    ).set_index(
        'evse_id', 
    )

    # Encode EVSEs' port type
    df_meta.loc[:, 'evse_type'] = df_meta['Port_Type'].str.split(' ').apply(lambda l: int(l[-1]))
    
    # Construct the EVSEs' full address
    df_meta['Site'] = (df_meta.Address + ', ' + df_meta.City + ', ' + df_meta.State_Province + ', ' + df_meta.Zip_Postal_Code.astype(str))

    # Get chargers' location (via Nominatim API)
    geolocator = Nominatim(user_agent="boulder-dataset-locations")
    
    df_meta_site_address, df_meta_site_location = {}, {}
    for site in df_meta.Site.unique():
        try: 
            site_geocode = geolocator.geocode(site, timeout=13)

            df_meta_site_address[site] = site_geocode.address
            df_meta_site_location[site] = shp.geometry.Point(site_geocode.longitude, site_geocode.latitude) # EPSG:4326

        except AttributeError:
            print(f'No coordinates found for: {site}. Please search manually...')
            df_meta_site_location[site] = None

    df_meta = pd.concat(
        [
            df_meta,
            df_meta.Site.map(df_meta_site_address).rename('address'),
            df_meta.Site.map(df_meta_site_location).rename('geometry')
        ],
        axis=1,
    )

    # Convert to GeoDataFrame
    df_meta = gpd.GeoDataFrame(df_meta, geometry=df_meta.geometry, crs=4326)

    # The Boulder dataset does not include EVSEs' max. power output information out-of-the-box. 
    # Therefore, we need to search for this particular information manually...
    # ```
    # A typical 7.2 kW home charger can recharge many EVs from empty to full in about 8 to 10 hours. 
    # Public stations often run at the higher end of the Level 2 scale, offering faster turnaround times.
    # ```
    # Since this value cannot be determined accurately per EVSE, to be on the "safe side", 
    # we choose to divide by 7.2 kW, which is compatible the the majority of Level 2 EVSEs.
    # Sources:
    #   - How Many Watts Does an Electric Car Charger Use?, https://everrati.com/blog/how-many-watts-electric-car-charger-use
    #   - Electric Vehicle Charging Stations, https://bouldercolorado.gov/services/electric-vehicle-charging-stations
    df_meta.loc[:, 'power_output_kW'] = 7.2

    # Visualize EVCS Locations    
    fig, ax = plt.subplots(1,1, figsize=(20, 20/1.618))

    mbb = df_meta.to_crs(3857).total_bounds
    max_width = max(mbb[2]-mbb[0], mbb[3]-mbb[1]) + 1_500
    remainder_width, remainder_height = abs(max_width - (mbb[2] - mbb[0])), abs(max_width - (mbb[3] - mbb[1]))

    # "Level 2" -> 2 (for visualization purposes)
    df_meta.to_crs(3857).astype({'power_output_kW':str}).plot(column='power_output_kW', ax=ax, markersize=df_meta['evse_type'] * 60, cmap='RdYlGn_r', legend=True, legend_kwds={'loc':'upper center', 'ncol':3, 'frameon':True, 'fancybox':True, 'markerscale':1.5, 'title_fontsize':17, 'fontsize':17})

    ax.set_xlim(mbb[0]-remainder_width/2, mbb[2]+remainder_width/2)
    ax.set_ylim(mbb[1]-remainder_height/2, mbb[3]+remainder_height/2)

    ctx.add_basemap(ax=ax, source=ctx.providers.CartoDB.Positron, attribution='')

    ax_leg = ax.get_legend()
    ax_leg.set_title('Power Output (kW)')

    plt.grid(False)
    plt.axis(False)

    plt.savefig(f'./data/fig/boulder_evcs_locations.png', dpi=300, bbox_inches='tight')

    # Save Results
    df_meta.to_pickle(f'./data/pkl/boulder_data.metadata.v3.pickle')
