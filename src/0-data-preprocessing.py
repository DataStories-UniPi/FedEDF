# %%
import pandas as pd
import numpy as np

# import pdb
import os
import glob
import argparse

from tqdm import tqdm
import matplotlib.pyplot as plt

import prep_utils


CFG_ROOT, CFG_EPS = './data', 1e-9
DUNDEE_DATASET_FEATURES = ['Charging event', 'CP ID', 'Connector', 'Start Date', 'Start Time', 'End Date', 'End Time', 'Total kWh', 'Cost', 'Site', 'Group', 'Model']
PORTO_DATASET_FEATURES = ['session_id', 'evse_id', 'user_id', 'start_time', 'end_time', 'total_energy_transfered', 'duration']
BOULDER_DATASET_FEATURES = ['ObjectId', 'Station_Name', 'Start_Date___Time', 'End_Date___Time', 'Energy__kWh_']
PALOALTO_DATASET_FEATURES = ['Station Name', 'Start Date', 'End Date', 'Energy (kWh)']


# In[9]:
if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='Centralized ("Share-All") VRF Worker')
    parser.add_argument('--dataset', help='Data source (default: "dundee")', default='dundee', choices=['dundee', 'porto', 'boulder', 'paloalto'], type=str, required=False)
    parser.add_argument('--sr_freq', help='Resampling Frequency (default: 1H)', default='1H', type=str, required=False)
    parser.add_argument('--min_pts', help='Minimum number of transactions for constructing EVSE timeseries (default:20 points)', default=20, type=int, required=False)
    parser.add_argument('--njobs', help='#CPUs (default:-1)', default=-1, type=int, required=False)
    args = parser.parse_args()


    if args.dataset == 'dundee':
        # Read Dundee Dataset
        df = pd.concat([
            pd.read_csv(
                src_dir, skipinitialspace=True, usecols=DUNDEE_DATASET_FEATURES
            ) for src_dir in glob.glob(os.path.join(CFG_ROOT, 'ev-load-open-data', '1. Input Data', '1. Dundee', '*.csv'))
        ], ignore_index=True).rename(
            {
                'Total kWh':'total_kWh',
                'Charging event':'session_id',
                'CP ID':'evse_id',
            }, 
            axis=1
        )

        # Preprocessing I
        #   # Drop NaN Values (based on ```total_kWh```)
        df.dropna(subset=['total_kWh'], inplace=True)

        #   # Drop 'cost' column, since it consists of - mostly - zeros
        df.drop('Cost', axis=1, inplace=True)
        
        #   # Parse inconsistent date format
        df['Start Date'], df['End Date'] = prep_utils.dundee_date_parser(df, 'Start Date'), prep_utils.dundee_date_parser(df, 'End Date')

        #   # Create the complete timestamp from the date and time columns
        df.loc[:, 'start_time'], df.loc[:, 'end_time'] = pd.to_datetime(
            df['Start Date'].dt.date.apply(str) + ' ' + df['Start Time']
        ), pd.to_datetime(
            df['End Date'].dt.date.apply(str) + ' ' + df['End Time']
        )

        df = prep_utils.dundee_inconsistent_dates(df.copy())

        #   # Calculate the duration of each session (in minutes)
        df.loc[:, 'duration'] = (df['end_time'] - df['start_time']).dt.total_seconds() / 60

        
    if args.dataset == 'porto':
        # Read Porto Dataset
        df = pd.concat(
            [
                pd.read_json(
                    src_dir
                ).drop(
                    [
                        'max_power', 
                        'min_power', 
                        'warnings'
                    ], 
                    axis=1
                ) for src_dir in glob.glob(os.path.join(CFG_ROOT, 'ev-load-inesc-tec', 'response_*_202*.json'))
            ], 
            ignore_index=True
        )

        # Preprocessing I
        # Drop NaN values @ total energy transfered
        df.dropna(subset=['total_energy_transfered'], inplace=True)
        
        # Fill NaN values @ timestamp using the latest recorded state
        df.end_time = df.end_time.fillna(
            pd.to_datetime(
                df.states.apply(
                    lambda l: l[-1]['datetime']
                )
            )
        )

        # Drop 'states' column, since it has no other usability
        df.drop('states', axis=1, inplace=True)

        # Convert ```total_energy_transfered``` from Wh to kWh
        df.loc[:, 'total_kWh'] = df.total_energy_transfered / 1000
        # Feature Engineering I
        # Calculate the duration of each session (in minutes)
        df.loc[:, 'duration'] = (df['end_time'] - df['start_time']).dt.total_seconds() / 60
        

    if args.dataset == 'boulder':
        df = pd.read_csv(
            os.path.join(CFG_ROOT, 'ev-load-open-data', '1. Input Data', '3. Boulder', 'ev_chargingstationdata_Boulder_March 2021.csv'),
            usecols=BOULDER_DATASET_FEATURES,
            parse_dates=['Start_Date___Time', 'End_Date___Time']
        )

        df.rename(
            {
                'ObjectId':'session_id',
                'Station_Name':'evse_id',
                'Start_Date___Time':'start_time',
                'End_Date___Time':'end_time',
                'Energy__kWh_':'total_kWh'
            },
            axis=1,
            inplace=True
        )

        # Feature Engineering I
        # Calculate the duration of each session (in minutes)
        df.loc[:, 'duration'] = (df['end_time'] - df['start_time']).dt.total_seconds() / 60
        

    if args.dataset == 'paloalto':
        df = pd.read_csv(
            os.path.join(CFG_ROOT, 'ev-load-open-data', '1. Input Data', '4. City of Palo Alto', 'ChargePoint Data CY20Q4.csv'),
            usecols=PALOALTO_DATASET_FEATURES,
        )

        # Add session identifier - nothing special, just an increasing number
        df.loc[:, 'session_id'] = df.index+1

        # Parse the starting timestamp for the Palo Alto dataset 
        df.loc[:, 'start_time'] = pd.to_datetime(df['Start Date'])

        # Parse the ending timestamp for the Palo Alto dataset 
        # NOTE: There are two distinct formats, namely, Plain text and Excel serial date.
        # 1. Create the date for the plain text (easy part)
        df.loc[:, 'end_time'] = pd.to_datetime(df['End Date'], errors='coerce')

        # 2. Fill NaT values with the parsed Excel serial dates (tricky part)
        df['end_time'] = df['end_time'].fillna(
            pd.to_datetime('1899-12-30') + pd.to_timedelta(
                pd.to_numeric(df['End Date'], errors='coerce'), 
                unit='D'
            )
        )
        
        # Rename columns for consistency
        df.rename(
            {
                'Station Name':'evse_id', 
                'Energy (kWh)':'total_kWh'
            },
            axis=1,
            inplace=True
        )

        # Feature Engineering I
        # Calculate the duration of each session (in minutes)
        df.loc[:, 'duration'] = (df['end_time'] - df['start_time']).dt.total_seconds() / 60


    # Apply filters
    df = df.loc[
        # Less than 2 day(s)
        (df['duration'].between(0, 2880, inclusive='right')) &
        # Drop "transactions" with energy above 200 kWh, as well as "transactions" with negative energy (Vehicle-2-Grid, maybe?)  
        (df['total_kWh'].between(0, 200, inclusive='neither'))  
    ]

    # Save processed dataset (transactional format) to pickle
    df.sort_values('start_time').to_pickle(
        os.path.join(CFG_ROOT, 'pkl', f'{args.dataset}_data.raw.pickle')
    )

    # Preprocessing II
    #   # Prune timeseries to a minimum of __MIN_PTS__ points
    no_of_sessions = df.groupby('evse_id').size() 
    df = df.loc[
        df['evse_id'].isin(
            no_of_sessions.loc[no_of_sessions > args.min_pts].index
        )
    ]

    #   # Convert transactions to timeseries
    tqdm.pandas()
    df.loc[:, 'time_axis'] = df.progress_apply(
        lambda l: pd.date_range(
            l['start_time'].floor('1H'), 
            l['end_time'].ceil('1H'), 
            freq='1H', 
            inclusive='left'
        ), 
        axis=1
    )

    #   # Convert kWh to kW
    df.loc[:, 'avg_kW'] = df['total_kWh'] / (
        (df['end_time'] - df['start_time']).dt.total_seconds()/3600
    )

    #   # Distribute across temporal intervals 
    df.loc[:, 'time_axis_weights'] = df.apply(lambda l: pd.Series(
            [l['start_time'], *l['time_axis'][1:], l['end_time']],
        ).sort_values(
        ).diff(
        ).dropna(
        ).dt.seconds.values / 3600, axis=1
    )

    df.loc[:, f'kW_{args.sr_freq}'] = df.time_axis_weights * df['avg_kW']
    df.loc[:, f'time_elapsed_{args.sr_freq}'] = (
        (df.time_axis_weights / df.time_axis_weights.apply(sum)) * df['duration']
    ).apply(
        np.cumsum
    )   # Get time elapsed per transaction and time bucket in minutes

    #   # Resample timeseries to desired frequency (e.g., 1H)
    # pdb.set_trace()
    df_explode = df.explode(
        ['time_axis', 'time_axis_weights', f'kW_{args.sr_freq}', f'time_elapsed_{args.sr_freq}']
    ).set_index(
        'time_axis'
    )
    
    df_evse_duration = df_explode.groupby(
        ['evse_id', 'session_id']
    ).resample(
        args.sr_freq
    ).agg(
        {
            f'time_elapsed_{args.sr_freq}': 'last'
        }
    ).groupby(
        ['evse_id', 'time_axis']
    ).mean()

    df_evse_demand = df_explode.groupby(
        'evse_id'
    ).resample(
        args.sr_freq
    ).agg(
        {
            'session_id':'nunique', 
            f'kW_{args.sr_freq}':'sum',
        }
    ).join(
        df_evse_duration
    ).rename(
        {
            'session_id': 'no_of_sessions',
            f'time_elapsed_{args.sr_freq}':'charging_time',
        }, 
        axis=1
    ).astype(
        {
            f'no_of_sessions':int,
            f'kW_{args.sr_freq}':float,
            f'charging_time':float
        }
    )

    #   # Visualize Results
    fig, ax = plt.subplots(9,8, figsize=(30, 30))

    for ((name, group), ax_i) in tqdm(zip(df_evse_demand.groupby(level=0), ax.flatten()), desc='Visualizing EVCS data...'):
        group.reset_index(level=0, drop=True).plot(drawstyle='steps-mid', ax=ax_i, legend=True)
        ax_i.set_title(f'EV Charger ID: {name}')

    plt.tight_layout()
    plt.savefig(
        os.path.join(
            CFG_ROOT, 'fig', f'{args.dataset}_data.demand_{args.sr_freq}.v4.png'
        ),
        dpi=500
    )

    # Save (plain) EVCS demand dataset
    df_evse_demand.to_pickle(
        os.path.join(CFG_ROOT, 'pkl', f'{args.dataset}_data.demand_{args.sr_freq}_{args.min_pts}_points.v4.pickle')
    )

