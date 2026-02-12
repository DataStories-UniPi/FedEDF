import os
import argparse
import pandas as pd

import sklearn
sklearn.set_config(transform_output='pandas')

from sklearn.metrics import r2_score
from sktime.performance_metrics.forecasting import (
    MeanAbsoluteScaledError, 
    MeanAbsolutePercentageError, 
    MeanSquaredError, 
    MeanAbsoluteError
)

import pmdarima as pmd

import stat_models as statml
import dataset as ds
import helper as hl


# %%
DATA_PATH = os.path.join('.', 'data')
EPS = 1e-1

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='Training statistical models')
    parser.add_argument('--dataset', help='Data source (default: "dundee")', default='dundee', choices=['dundee', 'porto', 'boulder', 'paloalto'], type=str, required=False)
    parser.add_argument('--sr_freq', help='Resampling Frequency (default: 1H)', default='1H', type=str, required=False)
    parser.add_argument('--min_pts', help='Minimum number of transactions for constructing EVSE timeseries (default:20 points)', default=20, type=int, required=False)
    parser.add_argument('--overwrite', help='Re-execute experiments (default: False)', action='store_true', required=False)
    args = parser.parse_args()

    # %% [markdown]
    # # Loading Enriched Dundee Dataset
    df_evse_demand = pd.read_pickle(
        os.path.join(
            DATA_PATH, 'pkl', 
            f"{args.dataset}_data.demand_{args.sr_freq}_{args.min_pts}_points.enriched.v4.pickle"
        )
    )

    # Scale charging time by max amount of duration (c.f., preprocessing)
    df_evse_demand['charging_time'] = df_evse_demand['charging_time'] / 2880  # 2280 --> 48hrs in minutes
   
    # Set forecasting target
    df_evse_demand.loc[:, 'power_next'] = df_evse_demand.groupby(level=1)['power_curr'].shift(-1)

    # Drop NaN values, caused by the above shift...
    df_evse_demand.dropna(subset=['power_next'], inplace=True)

    # %% [markdown]
    # # Split Dataset to Train/Dev/Test sets
    SARIMAX_FEATS = [
        # 'month_sin', 'month_cos',
        'week_sin', 'week_cos', 
        'day_sin', 'day_cos',
        'hour_sin', 'hour_cos',
        # 
        'activity_score',
        'no_of_sessions',
        'charging_time',
    ] 

    df_evse_demand_dates = pd.Series(df_evse_demand.index.get_level_values(2)).dt.date.sort_values().unique()
    df_evse_demand_dates_train_ix, df_evse_demand_dates_dev_ix, df_evse_demand_dates_test_ix = ds.timeseries_train_test_split(df_evse_demand_dates, dev_size=0.2, test_size=0.1, stratify=None, shuffle=False)

    print(f'Train \t@{(min(df_evse_demand_dates[df_evse_demand_dates_train_ix]), max(df_evse_demand_dates[df_evse_demand_dates_train_ix]))=};'+\
    f'\nDev \t@{(min(df_evse_demand_dates[df_evse_demand_dates_dev_ix]), max(df_evse_demand_dates[df_evse_demand_dates_dev_ix]))=};'+\
    f'\nTest \t@{(min(df_evse_demand_dates[df_evse_demand_dates_test_ix]), max(df_evse_demand_dates[df_evse_demand_dates_test_ix]))=}')

    X = df_evse_demand[[*SARIMAX_FEATS, 'power_next']].reset_index(level=(1,2))

    train_ix, dev_ix, test_ix = X.loc[X['timestamp'].dt.date.isin(df_evse_demand_dates[df_evse_demand_dates_train_ix])].index,\
                                X.loc[X['timestamp'].dt.date.isin(df_evse_demand_dates[df_evse_demand_dates_dev_ix])].index,\
                                X.loc[X['timestamp'].dt.date.isin(df_evse_demand_dates[df_evse_demand_dates_test_ix])].index

    X_train, X_dev, X_test = X.loc[train_ix].copy(), X.loc[dev_ix].copy(), X.loc[test_ix].copy()
    # pdb.set_trace()


    # %% [markdown]
    # # Train ARIMA/SARIMA/SARIMAX models
    # Source: https://medium.com/@tirthamutha/time-series-forecasting-using-sarima-in-python-8b75cd3366f2
    model_kwargs = {
        'ARIMA': {
            'model': pmd.arima.AutoARIMA,
            'model_params': dict(start_p=1, start_q=1, test='adf', seasonal=False, trace=True),
        },
        'SARIMA': {
            'model': pmd.arima.AutoARIMA,
            'model_params': dict(start_p=1, start_q=1, test='adf', m=12, seasonal=True, trace=True),
        },
        'SARIMAX': {
            'model': pmd.arima.AutoARIMA,
            'model_params': dict(start_p=1, start_q=1, test='adf', m=12, seasonal=True, trace=True),
            'X_feats': SARIMAX_FEATS
        }, 
    }

    model_instances = {}
    model_results = {}

    for key, model_param in model_kwargs.items():
        print(f'Training {key} models...')    
        models_dest_path = f'./data/pkl/{args.dataset}_data.demand_{args.sr_freq}.EVSE_{key}_model.{min(df_evse_demand_dates[df_evse_demand_dates_train_ix])}_{max(df_evse_demand_dates[df_evse_demand_dates_train_ix])}.v4.pickle'
        
        if not os.path.isfile(models_dest_path) or args.overwrite:
            models = statml.stat_model_fit(
                X_train,
                time_name='timestamp',
                oid_name='oid',
                target_name='power_next',
                model_name=f'{key}_instance',
                **model_kwargs[key],
                n_jobs=5
            )

            models.to_pickle(
                models_dest_path
            )
        else:
            models = pd.read_pickle(models_dest_path)
            
        # pdb.set_trace()
        model_results[key] = statml.stat_model_predict(
            X_test, 
            models, 
            f'{key}_instance',
            time_name='timestamp', 
            target_name='power_next',
            oid_name='oid',
            **model_kwargs[key]
        ).clip(
            lower=0
        )

        model_instances[key] = models

    # %% [markdown]
    # # Save Predictions
    series_test_preds = X_test.set_index(
        ['oid', 'timestamp']
    ).join(
        pd.concat(
            model_results,
            axis=1
        )
    ).dropna(
        # Get predictions only for eligible EVSEs (i.e., have a trained model)
        subset=[
            'ARIMA',
            'SARIMA',
            'SARIMAX'
        ]
    )

    series_test_preds.to_pickle(f'./data/pkl/{args.dataset}_data.demand_{args.sr_freq}.EVSE_{"_ALL_BASELINES_"}_model.{min(df_evse_demand_dates[df_evse_demand_dates_train_ix])}_{max(df_evse_demand_dates[df_evse_demand_dates_train_ix])}.v4.pickle')

    # %%
    model_results_metrics = hl.evaluate_predictions(
        series_test_preds.dropna(
            subset=['power_next']
        ),
        y_true_name='power_next',
        y_pred_names=list(model_results.keys()),
        eval_funs=[
            ('MASE_pct', MeanAbsoluteScaledError(sp=24), {'y_train':X_train['power_next'], 'oid_indices':X_train.groupby('oid', observed=False).groups}),
            ('SMAPE_pct', MeanAbsolutePercentageError(symmetric=True), {}),
            ('MAAPE_rads', hl.mean_arctangent_absolute_percentage_error, {}),
            ('WAPE_pct', hl.wape, {}),
            ('RMSE_kW', MeanSquaredError(square_root=True), {}),
            ('MAE_kW', MeanAbsoluteError(), {}),
            ('R2', r2_score, {}),
        ]
    )

    print(
        model_results_metrics.groupby(level=0).describe().T.loc[
            pd.IndexSlice[:, ['mean', '25%', '50%', '75%']], :
        ].round(2)
    )
    model_results_metrics.to_pickle(
        f'./data/pkl/{args.dataset}_data.demand_{args.sr_freq}.EVSE_{"_ALL_BASELINES_"}_model.{min(df_evse_demand_dates[df_evse_demand_dates_test_ix])}_{max(df_evse_demand_dates[df_evse_demand_dates_test_ix])}.metrics.v4.pickle'
    )
