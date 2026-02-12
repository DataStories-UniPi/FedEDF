import pandas as pd
import numpy as np


def dundee_date_parser(df,feat):    
    mask = df[feat].str.replace('\s+', '', regex=True).str.split('/').apply(len) > 1

    # Handle datetime inconsistency in the Dundee dataset
    dates = pd.concat((
        pd.to_datetime(
            df.loc[mask, feat], format='%d/%m/%Y', dayfirst=True
        ),
        pd.to_datetime(
            df.loc[~mask, feat], format='ISO8601'
        )
    )).sort_index()
    
    return dates


def dundee_inconsistent_dates(df):
    #   # Calculate the duration of each session (in minutes)
    df.loc[:, 'duration'] = (df['end_time'] - df['start_time']).dt.total_seconds() / 60

    #   # It sounds like some datetimes are inconsistent (have the month last, instead of the day)... 
    #   #   Start by flipping the date/month places of the ending dates...
    mask = df.duration < 0

    end_timestamps_flipped = pd.to_datetime(
        df.loc[mask, 'End Date'].astype(str), 
        format='%Y-%d-%m',
        errors='coerce'
    ).fillna(
        df.loc[mask, 'End Date']
    )

    mask_end_timestamps_flipped = (end_timestamps_flipped - df.loc[mask, 'Start Date']) > pd.Timedelta(seconds=0)
    df.loc[df.loc[mask].loc[mask_end_timestamps_flipped].index, 'End Date'] = end_timestamps_flipped[mask_end_timestamps_flipped]    

    #   #   ...for the remainder timestamps, flip the date/month places of the starting dates...
    start_timestamps_flipped = pd.to_datetime(
        df.loc[df.loc[mask].loc[~mask_end_timestamps_flipped].index, 'Start Date'].astype(str),
        format='%Y-%d-%m', errors='coerce'
    ).fillna(
        df.loc[df.loc[mask].loc[~mask_end_timestamps_flipped].index, 'Start Date']
    )

    mask_start_timestamps_flipped = (df.loc[df.loc[mask].loc[~mask_end_timestamps_flipped].index, 'End Date'] - start_timestamps_flipped) > pd.Timedelta(seconds=0)
    df.loc[df.loc[mask].loc[~mask_end_timestamps_flipped].loc[mask_start_timestamps_flipped].index, 'Start Date'] = start_timestamps_flipped[mask_start_timestamps_flipped]

    #   #   ...it appears that some timestamps are still erroneous... Flip their starting/ending times...
    mask_start_end_times = df.loc[mask].loc[~mask_end_timestamps_flipped].loc[~mask_start_timestamps_flipped].index 
    df.loc[mask_start_end_times, 'End Time'], df.loc[mask_start_end_times, 'Start Time'] = df.loc[mask_start_end_times, 'Start Time'], df.loc[mask_start_end_times, 'End Time']

    #   #   ...finally, there is one timestamp remaining, that its end was at 1970(!). Considering that only 7 kWh were consumed, we can assume that the charging process was completed in the same day
    # df.loc[df['End Date'].dt.year == 1970, 'End Date'] = df.loc[df['End Date'].dt.year == 1970, 'Start Date']

    #   # Re-Create the complete timestamp from the date and time columns
    df.loc[:, 'start_time'], df.loc[:, 'end_time'] = pd.to_datetime(
        df['Start Date'].dt.date.apply(str) + ' ' + df['Start Time']
    ), pd.to_datetime(
        df['End Date'].dt.date.apply(str) + ' ' + df['End Time']
    )
    
    return df


def extrapolate_energy_demand(vals, freq, steps_ahead=1, degree=1):
    """
    Extrapolate future energy demand using polynomial regression.

    Parameters:
    - vals: List or array of historical energy demand values.
    - freq: String frequency (e.g., 'H', 'D') used for extrapolation steps.
    - steps_ahead: How many future steps to predict (default 1).
    - degree: Degree of polynomial to fit (default 1 for linear).

    Returns:
    - Numpy array of predicted values (clamped to min 0).
    """
    vals_arr = np.array(vals)
    
    if len(vals_arr.shape) > 1:
        vals_arr = vals_arr.squeeze(axis=-1)

    total_points = len(vals_arr) + steps_ahead

    x = pd.date_range(start=0, periods=total_points, freq=freq, inclusive='left')
    x_unix = x.astype(int) // 10**9

    # Fit polynomial on historical portion
    fit = np.polyfit(x_unix[:len(vals_arr)], vals_arr, degree)
    model = np.poly1d(fit)

    # Predict next `steps_ahead` values
    future_times = x_unix[len(vals_arr):]
    predictions = model(future_times)

    # Clamp to min 0
    return np.maximum(predictions, 0)
