# %%
import pandas as pd
import geopandas as gpd
import numpy as np

from geopy.geocoders import Nominatim
import contextily as ctx
import shapely as shp

import os
import argparse

import matplotlib.pyplot as plt


CFG_ROOT, CFG_EPS = './data', 1e-9

# In[9]:
if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='Generate the metadata for the EVCS of the Dundee dataset')
    args = parser.parse_args()

    # Read Dundee Dataset
    df = pd.read_pickle(
        os.path.join(
            './data',
            'pkl',
            'dundee_data.raw.pickle'
        )
    )

    # Since the stations share common attributes, sort them in an immutable manner...
    df['Group'] = df['Group'].str.replace('\s+', '', regex=True).str.split(';').apply(lambda l: tuple(sorted(l)))

    # Create the Dundee metadata DataFrame (for future reference)
    df_meta = df.set_index('CP ID').loc[:, ['Site', 'Group', 'Model']].groupby(level=0).tail(1)

    #   # Get chargers' location (via Nominatim API)
    geolocator, df_meta_site_location = Nominatim(user_agent="dundee-dataset-locations"), {}

    for site in df_meta.Site.unique():
        try: 
            address=geolocator.geocode(site)
            df_meta_site_location[site] = shp.geometry.Point(address.longitude, address.latitude) # EPSG:4326
        except AttributeError:
            print(f'No coordinates found for: {site}. Please search manually...')
            df_meta_site_location[site] = None

    df_meta_site_location['Public Works Dept, Clepington Rd. Dundee'] = shp.geometry.Point(-2.9844468, 56.4780045)
    df_meta_site_location['Queen Street Car Park, Broughty Ferry, Dundee'] = shp.geometry.Point(-2.8728935, 56.4680155)
    df_meta_site_location['Social Work Building, Jack Martin Way, Dundee'] = shp.geometry.Point(-2.9468092, 56.4953655)
    df_meta_site_location['Olympia Multi-Storey Car Park, Dundee'] = shp.geometry.Point(-2.9625702, 56.4639562)
    df_meta_site_location['Greenmarket Multi Car Park, Dundee'] = shp.geometry.Point(-2.9712002, 56.4573000)
    df_meta_site_location['Marchbanks, Dundee'] = shp.geometry.Point(-3.0023612, 56.4746554)
    df_meta_site_location['Whitfield Centre, Dundee'] = shp.geometry.Point(-2.9170855, 56.4898305)
    df_meta_site_location['DCC Environment, 34 Harefield Road'] = shp.geometry.Point(-2.9979587, 56.4743229)
    df_meta_site_location['Brington Place Sheltered Housing, Dundee'] = shp.geometry.Point(-2.9206459, 56.4697854)
    df_meta_site_location['Janet Brougham House, Dundee'] = shp.geometry.Point(-2.9125024, 56.4767117)
    df_meta_site_location['Housing Office East, Dundee'] = shp.geometry.Point(-2.9355991, 56.4831255)
    df_meta_site_location['Housing Office West, Dundee'] = shp.geometry.Point(-3.0112717, 56.4725779)
    df_meta_site_location['Menziehill House, Dundee'] = shp.geometry.Point(-3.040243, 56.466218)
    df_meta_site_location['Oakland Day Centre, Dundee'] = shp.geometry.Point(-3.0044867, 56.4673273)
    df_meta_site_location['Turriff House Rannoch Road, Dundee'] = shp.geometry.Point(-3.003356, 56.4868865)
    df_meta_site_location['Earn Cresent, Dundee'] = shp.geometry.Point(-3.040259, 56.466213)
    df_meta_site_location['***TEST SITE*** Charge Your Car HQ'] = shp.geometry.Point(-2.9661151, 56.4621687)
    df_meta_site_location['Princes Street Charging Hub'] = shp.geometry.Point(-2.962651, 56.465149)
    df_meta_site_location['Sinclair Street'] = shp.geometry.Point(-3.0109801, 56.4720852)
    df_meta_site_location['South Tay Street'] = shp.geometry.Point(-2.976903, 56.459207)
    df_meta_site_location['Trades Lane'] = shp.geometry.Point(-2.965207, 56.462422)
    df_meta_site_location['Lochee Charging Hub, Dundee'] = shp.geometry.Point(-3.0111597, 56.4715570)
    
    df_meta = df_meta.set_index('Site', append=True).join(pd.Series(df_meta_site_location).rename('geometry').rename_axis('Site'), how='left')
    df_meta = gpd.GeoDataFrame(df_meta, geometry=df_meta.geometry, crs=4326)
    df_meta.reset_index(level=1, drop=False, inplace=True)

    df_meta.loc[:, 'power_output_kW'] = df_meta['Model'].map({
        "APT Triple Rapid Charger": 22,
        "APT 22kW Dual Outlet": 22,
        "APT 7kW Dual Outlet": 7,
        "APT 7kW Single Outlet": 7,
        "APT Dual Rapid Charger": 22,
        "APT 50kW Raption": 50    
    }).astype(int)

    df_meta.loc[:, 'power_outlets'] = df_meta['Model'].map({
        "APT Triple Rapid Charger": 3,
        "APT 22kW Dual Outlet": 2,
        "APT 7kW Dual Outlet": 2,
        "APT 7kW Single Outlet": 1,
        "APT Dual Rapid Charger": 2,
        "APT 50kW Raption": 2    
    }).astype(int)

    #   # Found a PDF with all EVCS in Scotland! Let's add the information for the "missing" (i.e., "***TEST SITE***..." locations)...
    #   # ...51547, 50912, "Lochee Charging Hub, Aimer Square, Dundee"
    df_meta.loc[[50912, 50913, 50914, 51547, 51548, 51549, 51550], 'Site'] = 'Lochee Charging Hub, Dundee'
    df_meta.loc[[50912, 50913, 50914, 51547, 51548, 51549, 51550], 'geometry'] = df_meta_site_location['Lochee Charging Hub, Dundee']

    #   # ...51426, 51429, 'Princes Street Charging Hub, Dundee'
    df_meta.loc[[51421, 51422, 51423, 51424, 51425, 51426, 51427, 51428, 51429], 'Site'] = 'Princes Street Charging Hub'
    df_meta.loc[[51421, 51422, 51423, 51424, 51425, 51426, 51427, 51428, 51429], 'geometry'] = df_meta_site_location['Princes Street Charging Hub']
    

    # Visualize EVSE Locations    
    fig, ax = plt.subplots(1,1, figsize=(20, 20/1.618))

    mbb = df_meta.to_crs(3857).total_bounds
    max_width = max(mbb[2]-mbb[0], mbb[3]-mbb[1]) + 1_500
    remainder_width, remainder_height = abs(max_width - (mbb[2] - mbb[0])), abs(max_width - (mbb[3] - mbb[1]))

    df_meta.to_crs(3857).astype({'power_output_kW':str}).plot(column='power_output_kW', ax=ax, markersize=df_meta['power_outlets'] * 60, cmap='RdYlGn_r', legend=True, legend_kwds={'loc':'upper center', 'ncol':3, 'frameon':True, 'fancybox':True, 'markerscale':1.5, 'title_fontsize':17, 'fontsize':17})

    ax.set_xlim(mbb[0]-remainder_width/2, mbb[2]+remainder_width/2)
    ax.set_ylim(mbb[1]-remainder_height/2, mbb[3]+remainder_height/2)

    ctx.add_basemap(ax=ax, source=ctx.providers.CartoDB.Positron, attribution='')

    ax_leg = ax.get_legend()
    ax_leg.set_title('Power Output (kW)')

    plt.grid(False)
    plt.axis(False)

    plt.savefig(f'./data/fig/dundee_evse_locations.png', dpi=300, bbox_inches='tight')


    # Save Results
    df_meta.to_pickle(f'./data/pkl/dundee_data.metadata.v3.pickle')
    