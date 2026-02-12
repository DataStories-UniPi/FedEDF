import json
import os
import numpy as np
import pandas as pd
from tqdm import tqdm

from collections import OrderedDict

from concurrent.futures import ThreadPoolExecutor
from joblib import Parallel, delayed

import multiprocessing
from sklearn.preprocessing import FunctionTransformer
from sktime.performance_metrics.forecasting import MeanAbsoluteScaledError

import xgboost as xgb

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import dataset as ds


def reset_cumsum(arr, condition):
    sumlm = np.frompyfunc(lambda a, b: a+b if condition(a,b) else b, 2, 1)
    return sumlm.accumulate(arr, dtype=int)


def get_parameters(model):
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def set_parameters(model, parameters):
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)


def weighted_sum(results, metric, **kwargs):
    # Get logger instance from **kwargs
    logger = kwargs.get('logger', None)

    # Weigh accuracy of each client by number of examples used
    results_metrics_aggregated, num_examples_aggregated = [], []

    for _, result in results:        
        # If the metric does not exist in the metrics dict sent by the client, 
        # proceed with the next client in the list
        if metric not in result.metrics:
            if logger is not None:
                logger.warn(f'[strategy][weighted_sum] Result does not contain {metric} metric!')
            continue

        # Fetch ```metric``` value
        result_metric = result.metrics[metric]

        # If the ```metric``` is string (i.e., jsonified list), deserialize the value
        if type(result_metric) is str:
            result_metric = np.array(json.loads(result_metric))

        # Finally, weight the result metric
        results_metrics_aggregated.append(result_metric * result.num_examples)
        num_examples_aggregated.append(result.num_examples)

    # Perform weighted sum and return result
    return np.sum(results_metrics_aggregated, axis=0) / np.sum(num_examples_aggregated, axis=0)


def applyParallel(df_grouped, fun, n_jobs=-1, **kwargs):
    '''
    Forked from: https://stackoverflow.com/a/27027632
    '''
    n_jobs = multiprocessing.cpu_count() if n_jobs == -1 else n_jobs
    print(f'Scaling {fun} to {n_jobs} CPUs')

    df_grouped_names = df_grouped.grouper.names
    _fun = lambda name, group: (fun(group.drop(df_grouped_names, axis=1)), name)

    result, keys = zip(*Parallel(n_jobs=n_jobs)(
        delayed(_fun)(name, group) for name, group in tqdm(df_grouped, **kwargs)
    ))
    return pd.concat(result, keys=keys, names=df_grouped_names)


def applyParallel_TPE(df_grouped, fun, n_jobs=-1, **kwargs):
    '''
    Scale pandas.DataFrame.groupby.apply queries using ThreadPoolExecutor.
    '''
    n_jobs = multiprocessing.cpu_count() if n_jobs == -1 else n_jobs
    print(f'Scaling {fun} to {n_jobs} CPUs')

    df_grouped_names = df_grouped.grouper.names

    def apply_fun(key_group):
        return key_group[0], fun(key_group[1])

    with ThreadPoolExecutor(max_workers=n_jobs) as executor:
        # Lazy execution while preserving order
        futures = [executor.submit(apply_fun, key_group) for key_group in tqdm(df_grouped)]
        results = dict(
            tqdm(
                (future.result() for future in futures),
                total=len(df_grouped),
                **kwargs
            )
        )

    return pd.concat(results.values(), keys=results.keys(), names=df_grouped_names)


def smape(y_true, y_pred, *args, **kwargs):
    '''
        Calculate Symmetric Mean Absolute Percentage Error (sMAPE)

        Parameters:
        y_true (array-like): Array of actual values
        y_pred (array-like): Array of forecasted values

        Returns:
        float: sMAPE value (%)
    '''
    numerator = np.abs(y_true - y_pred)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2

    # Calculate the sMAPE
    smape_value = np.mean(numerator / (denominator + 1e-9), axis=0)
    
    return smape_value


def wape(y_true, y_pred, *args, **kwargs):
    '''
        Calculate Weighted Absolute Percentage Error (WAPE)
        Source: https://medium.com/@vinitkothari.24/time-series-evaluation-metrics-mape-vs-wmape-vs-smape-which-one-to-use-why-and-when-part1-32d3852b4779

        Parameters:
        y_true (array-like): Array of actual values
        y_pred (array-like): Array of forecasted values

        Returns:
        float: WAPE value (%)
    '''

    numerator = np.abs(y_true - y_pred).sum(axis=0)
    denominator = y_true.sum(axis=0)

    # Calculate the WAPE
    wape_value = numerator / (denominator + 1e-9)
    return wape_value


def relative_absolute_error(y_true, y_pred):
    """
    Calculate Relative Absolute Error (RAE)
    
    RAE = sum(|y_true - y_pred|) / sum(|y_true - mean(y_true)|)
    
    Args:
        y_true: list or array of y_true values
        y_pred: list or array of y_pred values
    
    Returns:
        float: RAE value
    """
    actual_mean = sum(y_true) / len(y_true)
    
    numerator = sum(abs(a - p) for a, p in zip(y_true, y_pred))
    denominator = sum(abs(a - actual_mean) for a in y_true)
    
    return numerator / denominator


def mean_arctangent_absolute_percentage_error(actual, predicted):
    """
    Calculate Mean Arctangent Absolute Percentage Error (MAAPE)
    
    MAAPE = mean(arctan(|actual - predicted| / |actual|))
    
    Args:
        actual: list or array of actual values
        predicted: list or array of predicted values
    
    Returns:
        float: MAAPE value in radians
    """
    ape_values = []
    
    for a, p in zip(actual, predicted):
        if a == 0:
            # Handle division by zero case
            if p == 0:
                ape_values.append(0)
            else:
                ape_values.append(np.pi / 2)  # Maximum arctangent value
        else:
            ape = abs(a - p) / abs(a)
            ape_values.append(np.arctan(ape))
    
    return sum(ape_values) / len(ape_values)


def model_score(predictions, metric_fun, **kwargs):
    eps = kwargs.pop('eps', 0)
    y_true_column = predictions.columns[0]  # In the method the actual value is expected to be given first

    return pd.Series(
        data=[
            metric_fun(
                predictions[y_true_column].values + eps, 
                predictions[column].values + eps, 
                **kwargs
            ) for column in predictions.columns[1:]
        ],
        index=predictions.columns[1:]
    )


def evaluate_predictions(predictions, y_true_name, y_pred_names, eval_funs):
    models_to_compare = [y_true_name, *y_pred_names]
    eval_funs_res = {}

    for i, (fun_name, fun, fun_kwargs) in enumerate(eval_funs):
        
        if not isinstance(fun, MeanAbsoluteScaledError):
            eval_funs_res[f'{i}_{fun_name}'] = predictions.loc[:, models_to_compare].groupby(level=0).apply(
                lambda l: model_score(l, fun, **fun_kwargs)
            ) 
            continue

        y_train = fun_kwargs.pop('y_train', None)
        oid_indices = fun_kwargs.pop('oid_indices', None)

        eval_funs_res[f'{i}_{fun_name}'] = predictions.loc[:, models_to_compare].groupby(level=0).apply(
            lambda l: model_score(l, fun, **{
                'y_train': y_train[oid_indices[l.name]].values,
                **fun_kwargs
            })
        ) 
    
    return pd.concat(eval_funs_res)


def evse_energy_sequences(segment, time_axis, rnn_feats, fc_feats, y_feat):
    segment.sort_values(time_axis, inplace=True)
    
    # Set forecasting target
    segment.loc[:, 'power_next'] = segment[y_feat].shift(-1)

    return segment[[time_axis, *rnn_feats, *fc_feats, 'power_next']].dropna()


def evse_sequence_windowing(segment, time_axis, rnn_feats, fc_feats, y_feats, length_max=1024, length_min=20, stride=512):
    samples, misc, labels, time_axes = [], [], [], []
    
    rnn_feats_idx = [segment.columns.get_loc(input_feat) for input_feat in rnn_feats]
    fc_feats_idx = [segment.columns.get_loc(output_feat) for output_feat in fc_feats]
    y_feats_idx = [segment.columns.get_loc(output_feat) for output_feat in y_feats]
    time_axis_idx = segment.columns.get_loc(time_axis)
        
    for ptr_curr in range(0, len(segment), stride):
        segment_window = segment.iloc[ptr_curr:ptr_curr+length_max].copy()     

        if len(segment_window) < length_min:
            break

        samples.append(segment_window.iloc[:, rnn_feats_idx].values)
        misc.append(segment_window.iloc[-1, fc_feats_idx].values)
        labels.append(segment_window.iloc[-1, y_feats_idx].values)
        time_axes.append(segment_window.iloc[:, time_axis_idx].values)

    return pd.Series([samples, misc, labels, time_axes], index=['samples', 'misc', 'labels', 'time_axes'])


def evse_process_data_rnn(dataset_name, cluster_id, dataset_params, sequencing_params, windowing_params, logger):
    # Load EVSE Dataset
    evse_demand = pd.read_pickle(
        os.path.join(
            dataset_params['pickle_dir'],
            f'{dataset_name}_data.demand_{dataset_params["sr_freq"]}_{dataset_params["min_pts"]}_points',
            f'{dataset_name}_data.demand_{dataset_params["sr_freq"]}_{dataset_params["min_pts"]}_points.enriched.cluster_{cluster_id}.v4.pickle'
        )
    ).reset_index(level=(1,2))
    logger.info(f'[evse_process_data] Cluster ID #{cluster_id} - #Records: {len(evse_demand)}')

    # Feature Engineering II
    rolling__length, rolling__min_periods = 6, 4
    
    evse_demand.loc[:, f'{dataset_params["power_name"]}_std'] = evse_demand.groupby([dataset_params["object_name"]])[dataset_params["power_name"]].rolling(
        rolling__length, 
        min_periods=rolling__min_periods, 
        center=False
    ).std().reset_index(level=0, drop=True)

    # Exponential Average of Consumption
    evse_demand.loc[:, f'{dataset_params["power_name"]}_ema'] = evse_demand.groupby([dataset_params["object_name"]])[dataset_params["power_name"]].ewm(
        span=rolling__length, 
        min_periods=rolling__min_periods, 
        adjust=True
    ).mean().reset_index(level=0, drop=True)

    # Drop NaN values, caused by the rolling statistics
    evse_demand.dropna(inplace=True)

    # Split to train/dev/test split; 70/20/10%
    evse_dates = evse_demand['timestamp'].dt.date.sort_values().unique()
    train_time_axis_ix, dev_time_axis_ix, test_time_axis_ix = ds.timeseries_train_test_split(evse_dates, dev_size=0.2, test_size=0.1, stratify=None, shuffle=False)
        
    evse_demand.loc[evse_demand['timestamp'].dt.date.isin(evse_dates[train_time_axis_ix]), 'dataset_tr1_dev2_test3'] = 1
    evse_demand.loc[evse_demand['timestamp'].dt.date.isin(evse_dates[dev_time_axis_ix]),   'dataset_tr1_dev2_test3'] = 2
    evse_demand.loc[evse_demand['timestamp'].dt.date.isin(evse_dates[test_time_axis_ix]),  'dataset_tr1_dev2_test3'] = 3
    
    logger.info(f'[evse_process_data] Cluster ID #{cluster_id} - Sanity Check;\n\t{evse_demand.groupby([dataset_params["object_name"], "dataset_tr1_dev2_test3"])["timestamp"].is_monotonic_increasing.all()=}')
    logger.info(
        f'[evse_process_data] Cluster ID #{cluster_id} - '+\
        f'\nTrain @{(min(evse_dates[train_time_axis_ix]), max(evse_dates[train_time_axis_ix]))=};'+\
        f'\nDev @{(min(evse_dates[dev_time_axis_ix]), max(evse_dates[dev_time_axis_ix]))=};'+\
        f'\nTest @{(min(evse_dates[test_time_axis_ix]), max(evse_dates[test_time_axis_ix]))=}'
    )

    # Normalize / Scale Features
    # Scale charging time by max amount of duration (c.f., preprocessing)
    charging_time_norm_const = 2880     # 2280 --> 48hrs in minutes
    evse_demand['charging_time'] = evse_demand['charging_time'] / charging_time_norm_const

    # Normalize energy demand by maximum energy output of each EVSE, so that the energy will be around the range [0, 1]. 
    # Some stations will report values higher than their output, however that is expected, 
    # since we know - from domain experts - that EVSEs may exceed their nominal value by a factor of ±10%
    evse_max_energy_kwh = (evse_demand['power_output_kW'] * (pd.Timedelta(dataset_params["sr_freq"].lower()).total_seconds() / 3600)).values

    for col_name in evse_demand.columns:
        if dataset_params["power_name"] not in col_name:
            continue

        evse_demand.loc[:, f'{col_name}'] = evse_demand.loc[:, f'{col_name}'] / evse_max_energy_kwh

    # Normalize extrapolated energy demand
    evse_demand.loc[:, 'power_next_step1_extrap'] = evse_demand.loc[:, 'power_next_step1_extrap'] / evse_max_energy_kwh

    # Normalize Engineered Features 
    downtime_norm_const = evse_demand.loc[evse_demand.dataset_tr1_dev2_test3 == 1, f'downtime'].max()
    evse_demand.loc[:, f'downtime_scaled'] = evse_demand[f'downtime'] / downtime_norm_const

    no_of_sessions_norm_const = evse_demand.loc[evse_demand.dataset_tr1_dev2_test3 == 1, f'no_of_sessions'].max()
    evse_demand.loc[:, f'no_of_sessions_scaled'] = evse_demand[f'no_of_sessions'] / no_of_sessions_norm_const

    # Create EVSE fixed-length sequences
    # Parallelize pandas.DataFrame.groupby.apply query by using ProcessPoolExecutor
    evse_demand_seq = applyParallel_TPE(
        evse_demand.reset_index().groupby([dataset_params['object_name'], 'dataset_tr1_dev2_test3'], group_keys=True), 
        lambda l: evse_energy_sequences(l.copy(), **sequencing_params),
        n_jobs=dataset_params['njobs'],
        desc='Generating input features'
    )

    # Parallelize pandas.DataFrame.groupby.apply query by using ProcessPoolExecutor
    evse_demand_seq_windows = applyParallel_TPE(
        evse_demand_seq.reset_index().groupby([dataset_params['object_name'], 'dataset_tr1_dev2_test3'], group_keys=True),
        lambda l: evse_sequence_windowing(l, **windowing_params),
        n_jobs=dataset_params['njobs'],
        desc='Generating fixed-length sequences'
    ).reset_index(level=-1)\
    .pivot(columns=['level_2'])\
    .rename_axis([None, None], axis=1)\
    .sort_index(axis=1, ascending=False)

    evse_demand_seq_windows.columns = evse_demand_seq_windows.columns.droplevel(0)
    evse_demand_seq_windows = evse_demand_seq_windows.explode(['samples', 'misc', 'labels', 'time_axes'])

    # Save Results (for future reference)
    evse_demand_seq_windows.to_pickle(
        os.path.join(
            dataset_params['pickle_dir'],
            f'{dataset_name}_data.demand_{dataset_params["sr_freq"]}_sequences'+\
            f'_{len(windowing_params["rnn_feats"])}_rnn_inputs_{len(windowing_params["fc_feats"])}_fc_inputs_{len(windowing_params["y_feats"])}_outputs'+\
            f'_length_{windowing_params["length_max"]}_stride_{windowing_params["stride"]}.cluster_{cluster_id}.v4.pickle'
        )
    )
    evse_demand_seq_windows.dropna(inplace=True)

    logger.info(f'[evse_process_data] Cluster ID #{cluster_id} - '+\
        f'Train \t@{(evse_demand_seq_windows.xs(1, level=1).shape)=};'+\
        f'\nDev \t@{(evse_demand_seq_windows.xs(2, level=1).shape)=};'+\
        f'\nTest \t@{(evse_demand_seq_windows.xs(3, level=1).shape)=}'
    )

    return (
        evse_demand, 
        evse_demand_seq_windows, 
        {
            'charging_time_norm_const': charging_time_norm_const,
            'downtime_norm_const': downtime_norm_const,
            'no_of_sessions_norm_const': no_of_sessions_norm_const,
        }
    )


def evse_create_embeddings(dataset_name, dataset_params):
    # Create the Embedding Layer for EVSEs' Location
    location_tokens = pd.read_pickle(f'{dataset_params["pickle_dir"]}/{dataset_name}_data_{dataset_params["sr_freq"]}_building_tokens_v4.pkl')
    location_token_lookup = pd.read_pickle(f'{dataset_params["pickle_dir"]}/{dataset_name}_data_{dataset_params["sr_freq"]}_building_token_lookup_v4.pkl')
    location_embeddings = nn.Embedding(len(location_token_lookup), 15) 
    
    # Create the Embedding Layer for EVSEs' Model
    model_tokens = pd.read_pickle(f'{dataset_params["pickle_dir"]}/{dataset_name}_data_{dataset_params["sr_freq"]}_model_tokens_v4.pkl')
    model_token_lookup = pd.read_pickle(f'{dataset_params["pickle_dir"]}/{dataset_name}_data_{dataset_params["sr_freq"]}_model_token_lookup_v4.pkl')
    model_embeddings = nn.Embedding(len(model_token_lookup), 3) 

    return location_embeddings, model_embeddings, location_tokens, location_token_lookup, model_tokens, model_token_lookup


def evse_create_dataloader(dataset_params, evse_demand_seq_windows, location_tokens, location_token_lookup, model_tokens, model_token_lookup):
    # Create unified train/dev/test dataset(s)
    train_delta_windows = evse_demand_seq_windows[evse_demand_seq_windows.index.get_level_values(1) == 1].copy()
    dev_delta_windows = evse_demand_seq_windows[evse_demand_seq_windows.index.get_level_values(1) == 2].copy()
    test_delta_windows = evse_demand_seq_windows[evse_demand_seq_windows.index.get_level_values(1) == 3].copy()

    # Create EVSE features' temporal sequence (i.e. training dataset)
    identity_function = FunctionTransformer(None) # We do not need a scaler for now, so we use the Identity function...
    train_dataset = ds.EDFDataset_v2(
        location_tokens, model_tokens, 
        location_token_lookup, model_token_lookup, 
        train_delta_windows, 
        scaler=identity_function
    )

    dev_dataset, test_dataset = ds.EDFDataset_v2(
        location_tokens, model_tokens, 
        location_token_lookup, model_token_lookup, 
        dev_delta_windows, 
        scaler=train_dataset.scaler
    ), ds.EDFDataset_v2(
        location_tokens, model_tokens, 
        location_token_lookup, model_token_lookup, 
        test_delta_windows, 
        scaler=train_dataset.scaler
    )

    train_loader = DataLoader(train_dataset, batch_size=dataset_params['bs'], shuffle=True, collate_fn=train_dataset.pad_collate)
    dev_loader = DataLoader(dev_dataset, batch_size=dataset_params['bs'], shuffle=False, collate_fn=dev_dataset.pad_collate)
    test_loader = DataLoader(test_dataset, batch_size=dataset_params['bs'], shuffle=False, collate_fn=test_dataset.pad_collate)

    return train_loader, dev_loader, test_loader


def evse_process_data_xgboost(dataset_name, cluster_id, dataset_params, feat_eng_params, logger):
    # Load EVSE Dataset
    evse_demand = pd.read_pickle(
        os.path.join(
            dataset_params['pickle_dir'],
            f'{dataset_name}_data.demand_{dataset_params["sr_freq"]}_{dataset_params["min_pts"]}_points',
            f'{dataset_name}_data.demand_{dataset_params["sr_freq"]}_{dataset_params["min_pts"]}_points.enriched.cluster_{cluster_id}.v4.pickle'
        )
    ).reset_index(level=(1,2))
    logger.info(f'[evse_process_data] Cluster ID #{cluster_id} - #Records: {len(evse_demand)}')

    # Feature Engineering    
    #   1. Rolling Statistics 
    evse_demand.loc[:, f'{dataset_params["power_name"]}_std'] = evse_demand.groupby([dataset_params["object_name"]])[dataset_params["power_name"]].rolling(
        feat_eng_params['rolling__window'], 
        min_periods=feat_eng_params['rolling__min_periods'], 
        center=False
    ).std().reset_index(level=0, drop=True)

    evse_demand.loc[:, f'{dataset_params["power_name"]}_ema'] = evse_demand.groupby([dataset_params["object_name"]])[dataset_params["power_name"]].ewm(
        span=feat_eng_params['rolling__window'], 
        min_periods=feat_eng_params['rolling__min_periods'], 
        adjust=True
    ).mean().reset_index(level=0, drop=True)

    #   2. Lagged Features
    evse_demand = evse_demand.groupby(
        [dataset_params["object_name"]], group_keys=False, as_index=False
    ).apply(
        lambda l: l.assign(**{
            **{
                f'{dataset_params["power_name"]}_lag{lag}': evse_demand[dataset_params["power_name"]].shift(lag) for lag in [5, 24, 48]
            }, **{
                f'{dataset_params["power_name"]}_ema_lag{lag}': evse_demand[f'{dataset_params["power_name"]}_ema'].shift(lag) for lag in [5, 24, 48]
            }, **{
                f'{dataset_params["power_name"]}_std_lag{lag}': evse_demand[f'{dataset_params["power_name"]}_std'].shift(lag) for lag in [5, 24, 48]
            }
        })
    )

    #   3. Set forecasting target
    evse_demand.loc[:, 'power_next'] = evse_demand.groupby([dataset_params["object_name"]])[dataset_params["power_name"]].shift(-1)

    #   4. Drop NaN values, caused by the rolling statistics and/or lagged features
    evse_demand.dropna(inplace=True)

    # Split to train/dev/test split; 70/20/10%
    evse_dates = evse_demand[feat_eng_params['time_axis']].dt.date.sort_values().unique()
    train_time_axis_ix, dev_time_axis_ix, test_time_axis_ix = ds.timeseries_train_test_split(evse_dates, dev_size=0.2, test_size=0.1, stratify=None, shuffle=False)
                                
    train_ix, dev_ix, test_ix = evse_demand.loc[evse_demand[feat_eng_params['time_axis']].dt.date.isin(evse_dates[train_time_axis_ix])].index,\
                                evse_demand.loc[evse_demand[feat_eng_params['time_axis']].dt.date.isin(evse_dates[dev_time_axis_ix])].index,\
                                evse_demand.loc[evse_demand[feat_eng_params['time_axis']].dt.date.isin(evse_dates[test_time_axis_ix])].index

    evse_demand.loc[train_ix, 'dataset_tr1_dev2_test3'] = 1
    evse_demand.loc[dev_ix,   'dataset_tr1_dev2_test3'] = 2
    evse_demand.loc[test_ix,  'dataset_tr1_dev2_test3'] = 3

    logger.info(
        f'[evse_process_data] Cluster ID #{cluster_id} - '+\
        f'\nTrain @{(min(evse_dates[train_time_axis_ix]), max(evse_dates[train_time_axis_ix]))=};'+\
        f'\nDev @{(min(evse_dates[dev_time_axis_ix]), max(evse_dates[dev_time_axis_ix]))=};'+\
        f'\nTest @{(min(evse_dates[test_time_axis_ix]), max(evse_dates[test_time_axis_ix]))=}'
    )

    # Normalize / Scale Features
    # Scale charging time by max amount of duration (c.f., preprocessing)
    charging_time_norm_const = 2880     # 2280 --> 48hrs in minutes
    evse_demand['charging_time'] = evse_demand['charging_time'] / charging_time_norm_const

    # Normalize energy demand by maximum energy output of each EVSE, so that the energy will be around the range [0, 1]. 
    # Some stations will report values higher than their output, however that is expected, 
    # since we know - from domain experts - that EVSEs may exceed their nominal value by a factor of ±10%
    evse_max_energy_kw = (evse_demand['power_output_kW'] * (pd.Timedelta(dataset_params["sr_freq"].lower()).total_seconds() / 3600)).values

    for col_name in evse_demand.columns:
        if dataset_params["power_name"] not in col_name:
            continue

        evse_demand.loc[:, f'{col_name}'] = evse_demand.loc[:, f'{col_name}'] / evse_max_energy_kw

    # Normalize extrapolated energy demand
    evse_demand.loc[:, 'power_next_step1_extrap'] = evse_demand.loc[:, 'power_next_step1_extrap'] / evse_max_energy_kw

    # Normalize Engineered Features 
    downtime_norm_const = evse_demand.loc[train_ix, f'downtime'].max()
    evse_demand.loc[:, f'downtime_scaled'] = evse_demand[f'downtime'] / downtime_norm_const

    no_of_sessions_norm_const = evse_demand.loc[train_ix, f'no_of_sessions'].max()
    evse_demand.loc[:, f'no_of_sessions_scaled'] = evse_demand[f'no_of_sessions'] / no_of_sessions_norm_const

    #   5. Cast categorical dtypes
    evse_demand.oid = evse_demand.oid.astype('category')
    # evse_demand.power_outlets = evse_demand.power_outlets.astype('category')
    evse_demand.Site = evse_demand.Site.astype('category')
    evse_demand.power_output_kW = evse_demand.power_output_kW.astype('category')
    evse_demand.is_offline = evse_demand.power_output_kW.astype('bool')

    # Save Results (for future reference)
    evse_demand.to_pickle(
        os.path.join(
            dataset_params['pickle_dir'],
            f'{dataset_name}_data.demand_{dataset_params["sr_freq"]}_tabular'+\
            f'_{len(feat_eng_params["X_feats"])}_xgb_inputs_{len(feat_eng_params["y_feat"])}_outputs'+\
            f'.cluster_{cluster_id}.v4.pickle'
        )
    )

    # Get Input/Output features for XGBoost training
    X, y_reg = (
        evse_demand.loc[:, feat_eng_params['X_feats']].copy(), 
        evse_demand.loc[:, feat_eng_params['y_feat']].copy()
    )

    (
        X_train, y_train,
        X_dev, y_dev,
        X_test, y_test
    ) = (
        X.loc[train_ix, :].copy(), y_reg.loc[train_ix].copy(), 
        X.loc[dev_ix, :].copy(), y_reg.loc[dev_ix].copy(), 
        X.loc[test_ix, :].copy(), y_reg.loc[test_ix].copy(), 
    )

    return (
        {'X':X_train, 'y':y_train}, 
        {'X':X_dev, 'y':y_dev}, 
        {'X':X_test, 'y':y_test}, 
        {
            'charging_time_norm_const': charging_time_norm_const,
            'downtime_norm_const': downtime_norm_const,
            'no_of_sessions_norm_const': no_of_sessions_norm_const,
        }
    )


def xgb_tree_predictions(xgb_model, X, **kwargs):
    booster = xgb_model.get_booster()
    n_estimators = len(booster.get_dump())

    X_dm = xgb.DMatrix(X, **kwargs)

    tree_preds = []
    for tree_ix in range(n_estimators):
        tree_preds.append(
            booster.predict(X_dm, iteration_range=(tree_ix, tree_ix+1), output_margin=True)
        )

    return np.vstack(tree_preds).T  # <n_samples, n_estimators>


def fedxgbllr_cnn_create_dataloader(aggregated_trees, dataset, logger, **kwargs):
    X = []
    
    for client_id_tree, client_id in aggregated_trees:
        logger.info(f'[fedxgbllr_cnn_create_dataloader] Fetching the individual predictions of tree #{client_id}...')
        X.append(
            xgb_tree_predictions(
                client_id_tree, dataset['X'], **kwargs
            )
        )

    X, y = torch.from_numpy(
        np.expand_dims(
            np.concatenate(X, axis=1), 
            axis=-2
        )
    ), torch.from_numpy(
        np.expand_dims(
            np.array(dataset['y']),
            axis=-1
        )
    )

    return TensorDataset(X, y)


def save_data_parquet(federation, oid, train_set, dev_set, test_set, save_path):
    # Ensure ```save_path``` exists
    save_path.mkdir(parents=True, exist_ok=True)

    # Create train/dev/test directories
    (save_path_train := save_path / 'train').mkdir(parents=True, exist_ok=True)
    (save_path_dev := save_path / 'validation').mkdir(parents=True, exist_ok=True)
    (save_path_test := save_path / 'test').mkdir(parents=True, exist_ok=True)

    # Prepare dataset slices
    train_set_cat = pd.concat(train_set, axis=1)
    train_set_cat.columns = train_set_cat.columns.droplevel(0)
    
    dev_set_cat = pd.concat(dev_set, axis=1)
    dev_set_cat.columns = dev_set_cat.columns.droplevel(0)
    
    test_set_cat = pd.concat(test_set, axis=1)
    test_set_cat.columns = test_set_cat.columns.droplevel(0)

    # Save dataset slices
    train_set_cat.to_parquet(
        save_path_train / f'{federation}_{"train"}_h1_w6_v1_client{oid}.parquet.gzip', 
        engine='pyarrow'
    )
    dev_set_cat.to_parquet(
        save_path_dev / f'{federation}_{"validation"}_h1_w6_v1_client{oid}.parquet.gzip',
        engine='pyarrow'
    )
    test_set_cat.to_parquet(
        save_path_test / f'{federation}_{"test"}_h1_w6_v1_client{oid}.parquet.gzip',
        engine='pyarrow'
    )
