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
PALOALTO_DATASET_META_FEATURES = ['Station Name', 'Model Number', 'System S/N', 'MAC Address', 'Port Type', 'Port Number', 'Plug Type', 'Address 1', 'City', 'State/Province', 'Postal Code', 'Country', 'Longitude', 'Latitude']


# In[9]:
if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='Generate the metadata for the EVCS of the Boulder dataset')
    args = parser.parse_args()


    # Read Boulder Dataset
    df_meta = pd.read_csv(
        os.path.join(CFG_ROOT, 'ev-load-open-data', '1. Input Data', '4. City of Palo Alto', 'ChargePoint Data CY20Q4.csv'),
        usecols=PALOALTO_DATASET_META_FEATURES,
    ).rename(
        {
            'Station Name':'evse_id',
        },
        axis=1,
    ).groupby(
        'evse_id', as_index=False
    ).apply(
        lambda l: l.dropna(subset=['Model Number']).tail(1)
    ).set_index(
        'evse_id'
    )

    # Encode EVSEs' port type
    df_meta.loc[:, 'evse_type'] = df_meta['Port Type'].str.split(' ').apply(lambda l: int(l[-1]))
    
    # Construct the EVSEs' full address    
    df_meta['Site'] = (df_meta['Address 1'] + ', ' + df_meta['City'] + ', ' + df_meta['State/Province'] + ', ' + df_meta['Postal Code'].astype(str))

    # Palo Alto has all necessary metadata, except nominal power output. Searching for each serial, provides us with that info.
    # Except CT4020-HD and CT4020-HD-GW, with nominal electrical rating of 32A @ 208/240V per port [3], all others have a rating
    # of 30A @ 208/240V per port [1,2,4]. Thus all EVSEs have a nominal power output of 7.2kW, except the aforementioned two with
    # nominal power output of 7.68 kW per port.
    # Sources:
    # [1]: https://www.platt.com/p/2121344/chargepoint/comm-ev-charger-single-head-30a/cgtct4010hdgwlte
    # [2]: https://www.chargepoint.com/files/CT2100-Installation-Guide75-001020-01Rev4.0.pdf
    # [3]: https://www.gexpro.com/p/2004365/chargepoint/ev-charger-dual-head-240v-32a/ct4020-hd-gw-lte
    # [4]: https://ideadigitalasset.com/DAMRoot/Original/10001/10962_ID-BRO-v1-92600.pdf
    df_meta_power_output = {
        'CT4010-HD-GW': 7.2,
        'CT2100-HD-CDMA-CCR': 7.2, 
        'CT2100-HD-CCR': 7.2, 
        'CT4020-HD': 7.68,
        'CT4020-HD-GW': 7.68, 
        'CT2000-HD-GW1-CCR': 7.2, 
        'CT2000-HD-CCR': 7.2, 
        'CTHCR-S': 7.2,
        'CTHDR': 7.2, 
        'CTHDR-S': 7.2
    }

    df_meta.loc[:, 'power_output_kW'] = df_meta[
        'Model Number'
    ].map(
        df_meta_power_output
    )
    
    # Convert to GeoDataFrame
    df_meta.loc[:, 'geometry'] = df_meta[['Longitude', 'Latitude']].apply(
        lambda l: shp.geometry.Point(*l), axis=1
    )

    df_meta = gpd.GeoDataFrame(df_meta, geometry=df_meta.geometry, crs=4326)

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

    plt.savefig(f'./data/fig/paloalto_evcs_locations.png', dpi=300, bbox_inches='tight')

    # Save Results
    df_meta.to_pickle(f'./data/pkl/paloalto_data.metadata.v3.pickle')
