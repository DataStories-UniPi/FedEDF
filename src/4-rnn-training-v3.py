# %%
import pandas as pd
import numpy as np
import argparse
import os

# %%
import torch
from torch import nn
from torch.utils.data import DataLoader

torch.manual_seed(10)
torch.autograd.set_detect_anomaly(True)

# %%
from sklearn.preprocessing import FunctionTransformer

# %%
import dataset as ds 
import models as ml
import train as tr
import helper as hl


CFG_ROOT, CFG_EPS = './data', 1e-9


# In[9]:
if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='Centralized Energy Demand Forecasting (EDF) Worker')
    # GENERAL PARAMS
    parser.add_argument('--gpuid', help='GPU ID', default=0, type=int)
    parser.add_argument('--njobs', help='#CPUs', default=128, type=int)
    # RNN PARAMS
    parser.add_argument('--rnn_cell', help='Type of RNN Cell', type=str, default='lstm', choices=['lstm', 'gru'])
    parser.add_argument('--bi', help='Use Bidirectional RNN Cell', action='store_true')
    parser.add_argument('--hidden_size', help='RNN Hidden Size', type=int, default=350)
    parser.add_argument('--num_layers', help='RNN Number of Layers', type=int, default=1)
    parser.add_argument('--fc_layers', help='Number/Size of FC Layers', type=str, default='150')
    # DATA PARAMS
    parser.add_argument('--data', help='Select Dataset', choices=['dundee', 'porto', 'boulder', 'paloalto'], type=str, required=True)
    parser.add_argument('--sr_freq', help='Resampling Frequency (default: 12H)', default='12H', type=str)
    parser.add_argument('--min_pts', help='Minimum number of transactions for constructing EVSEs timeseries (default:100 points)', default=100, type=int)
    # TRAINING PARAMS
    parser.add_argument('--bs', help='Batch Size', default=1, type=int)
    parser.add_argument('--length', help='Rolling Window Length (default: 14)', default=14, type=int)
    parser.add_argument('--stride', help='Rolling Window Stride (default: 2)', default=2, type=int)
    parser.add_argument('--n_epochs', help='#Epochs for Model Training (default: 100)', default=100, type=int)
    parser.add_argument('--patience', help='Patience (#Epochs) for Early Stopping (default: 10)', default=10, type=int)
    # MISC PARAMS
    parser.add_argument('--skip_train', help='Skip training; Evaluate best model @ Test Set', action='store_true')
    args = parser.parse_args()


    # # Load Dataset
    df = pd.read_pickle(
        os.path.join(
            CFG_ROOT, 'pkl', 
            f"{args.data}_data.demand_{args.sr_freq}_{args.min_pts}_points.enriched.v4.pickle"
        )
    ).reset_index(level=(1,2))


    # Generate the Lookup table / Embedding layer of EVCS' location
    building_tokens = df.groupby(['oid',])['building'].unique().apply(lambda l: l[0])
    building_token_lookup = pd.Series({
        v:k for k, v in enumerate(building_tokens.sort_values().unique())
    })
    # building_embeddings = nn.Embedding(len(building_token_lookup), len(building_token_lookup)//2) 
    building_embeddings = nn.Embedding(len(building_token_lookup), 15) 

    # Generate the Lookup table / Embedding layer of EVCS' model
    model_tokens = df.groupby(['oid',])['model'].unique().apply(lambda l: l[0])
    model_token_lookup = pd.Series({
        v:k for k, v in enumerate(model_tokens.sort_values().unique())
    })
    # model_embeddings = nn.Embedding(len(model_token_lookup), len(model_token_lookup)//2) 
    model_embeddings = nn.Embedding(len(model_token_lookup), 3) 

    building_tokens.to_pickle(f'./data/pkl/{args.data}_data_{args.sr_freq}_building_tokens_v4.pkl')               # Saving chargers' location tokens for future reference
    building_token_lookup.to_pickle(f'./data/pkl/{args.data}_data_{args.sr_freq}_building_token_lookup_v4.pkl')   # Saving chargers' location lookup for future reference
    model_tokens.to_pickle(f'./data/pkl/{args.data}_data_{args.sr_freq}_model_tokens_v4.pkl')                     # Saving chargers' model           for future reference
    model_token_lookup.to_pickle(f'./data/pkl/{args.data}_data_{args.sr_freq}_model_token_lookup_v4.pkl')         # Saving chargers' model lookup    for future reference


    # Feature Engineering II
    rolling_params__length, rolling_params__min_periods = 6, 4
    
    df.loc[:, 'power_curr_std'] = df.groupby(["oid"])['power_curr'].rolling(
        rolling_params__length, 
        min_periods=rolling_params__min_periods, 
        center=False
    ).std().reset_index(level=0, drop=True)

    # Exponential Average of Consumption
    df.loc[:, 'power_curr_ema'] = df.groupby(["oid"])['power_curr'].ewm(
        span=rolling_params__length, 
        min_periods=rolling_params__min_periods, 
        adjust=True
    ).mean().reset_index(level=0, drop=True)

    df.dropna(inplace=True)


    # Split to train/dev/test split; 70/20/10%
    evse_dates = df['timestamp'].dt.date.sort_values().unique()
    train_dates, dev_dates, test_dates = ds.timeseries_train_test_split(evse_dates, dev_size=0.2, test_size=0.1, stratify=None, shuffle=False)
        
    print(
        f'Train \t@{(min(evse_dates[train_dates]), max(evse_dates[train_dates]))=};'+\
        f'\nDev \t@{(min(evse_dates[dev_dates]), max(evse_dates[dev_dates]))=};'+\
        f'\nTest \t@{(min(evse_dates[test_dates]), max(evse_dates[test_dates]))=}'
    )

    df.loc[df['timestamp'].dt.date.isin(evse_dates[train_dates]), 'dataset_tr1_dev2_test3'] = 1
    df.loc[df['timestamp'].dt.date.isin(evse_dates[dev_dates]),   'dataset_tr1_dev2_test3'] = 2
    df.loc[df['timestamp'].dt.date.isin(evse_dates[test_dates]),  'dataset_tr1_dev2_test3'] = 3

    print(f"Sanity Check #1;\n\t{df.groupby(['oid', 'dataset_tr1_dev2_test3'])['timestamp'].is_monotonic_increasing.all()=}")


    # Scale charging time by max amount of duration (c.f., preprocessing)
    df['charging_time'] = df['charging_time'] / 2880  # 2280 --> 48hrs in minutes

    # Normalize energy demand by maximum energy output of each EVSE, so that the energy will be around the range [0, 1]. 
    # Some stations will report values higher than their output, however that is expected, 
    # since we know - from domain experts - that EVSEs may exceed their nominal value by a factor of ±10%
    evse_max_energy_kwh = (df['power_output_kW'] * (pd.Timedelta(args.sr_freq.lower()).total_seconds() / 3600)).values

    for col_name in df.columns:
        if 'power_curr' not in col_name:
            continue

        df.loc[:, f'{col_name}'] = df.loc[:, f'{col_name}'] / evse_max_energy_kwh

    # Normalize extrapolated energy demand
    df.loc[:, 'power_next_step1_extrap'] = df.loc[:, 'power_next_step1_extrap'] / evse_max_energy_kwh

    # Normalize Engineered Features 
    downtime_norm_const = df.loc[df.dataset_tr1_dev2_test3 == 1, f'downtime'].max()
    df.loc[:, f'downtime_scaled'] = df[f'downtime'] / downtime_norm_const

    no_of_sessions_norm_const = df.loc[df.dataset_tr1_dev2_test3 == 1, f'no_of_sessions'].max()
    df.loc[:, f'no_of_sessions_scaled'] = df[f'no_of_sessions'] / no_of_sessions_norm_const

    # %%
    params = dict(
        time_axis='timestamp',
        rnn_feats=[
            'day_sin', 'day_cos',
            'hour_sin', 'hour_cos', 
            'week_sin', 'week_cos', 
            #
            'power_curr_logdelta',
            'power_next_step1_extrap',
            # 
            'power_curr_std', 
            'power_curr_ema',
            #
            f'downtime_scaled',
            #
            'no_of_sessions_scaled',
            'charging_time',
            'power_curr', 
        ],
        # Extra features to include in the Fully Connected (FC) layer (excl. ```building``` and ```model```)
        fc_feats=[
            'power_output_kW', 
        ], 
        y_feat='power_curr',
    )

    evcs_dataset = hl.applyParallel(
        df.reset_index().groupby(['oid', 'dataset_tr1_dev2_test3'], group_keys=True), 
        lambda l: hl.evse_energy_sequences(l.copy(), **params),
        n_jobs=args.njobs
    )


    # %%
    windowing_params = dict(
        length_min=args.length // 2, 
        length_max=args.length, 
        stride=args.stride, 
        time_axis='timestamp', 
        rnn_feats=params['rnn_feats'], 
        fc_feats=params['fc_feats'], 
        y_feats=['power_next'], 
    )
    print(f'{windowing_params["rnn_feats"]=}')
    print(f'{windowing_params["fc_feats"]=}')
    print(f'{windowing_params["y_feats"]=}')

    evcs_dataset_windows = hl.applyParallel(
        evcs_dataset.reset_index().groupby(['oid', 'dataset_tr1_dev2_test3']),
        lambda l: hl.evse_sequence_windowing(l, **windowing_params),
        n_jobs=args.njobs
    ).reset_index(level=-1)\
    .pivot(columns=['level_2'])\
    .rename_axis([None, None], axis=1)\
    .sort_index(axis=1, ascending=False)

    evcs_dataset_windows.columns = evcs_dataset_windows.columns.droplevel(0)
    evcs_dataset_windows = evcs_dataset_windows.explode(['samples', 'misc', 'labels', 'time_axes'])


    # Save Results (for future reference)
    evcs_dataset_windows.to_pickle(
        os.path.join(
            CFG_ROOT, 'pkl', 
            f"{args.data}_data.demand_{args.sr_freq}_sequences"+\
            f"_{len(windowing_params['rnn_feats'])}_rnn_inputs_{len(windowing_params['fc_feats'])}_fc_inputs_{len(windowing_params['y_feats'])}_outputs"+\
            f"_length_{windowing_params['length_max']}_stride_{windowing_params['stride']}.v4.pickle"
        )
    )

    print(
        f'Train \t@{(evcs_dataset_windows.xs(1, level=1).shape)=};'+\
        f'\nDev \t@{(evcs_dataset_windows.xs(2, level=1).shape)=};'+\
        f'\nTest \t@{(evcs_dataset_windows.xs(3, level=1).shape)=}'
    )

    print(evcs_dataset_windows.dropna(inplace=True))
    # pdb.set_trace()

    # %%
    #   * #### DATASET STATISTICS
    print(f'Points per EVSE {df.groupby(["oid"]).apply(len).describe()=}\n')
    print(f"Duration per EVSE {df.groupby(['oid']).agg({'timestamp':[np.min, np.max]}).diff(axis=1).loc[:, pd.IndexSlice[:, 'max']].describe()=}\n")
    print(f"Sampling Rate per EVSE {df.groupby(['oid']).timestamp.diff().describe()=}\n")
    print(f"Time-steps per Window {evcs_dataset_windows.samples.apply(lambda l: l.shape[0]).describe()=}\n")

    # %%
    # # Create PyTorch Data Loaders
    # # Create unified train/dev/test dataset(s)
    train_delta_windows, dev_delta_windows, test_delta_windows = evcs_dataset_windows.xs(
        1, level=1
    ).copy(), evcs_dataset_windows.xs(
        2, level=1
    ).copy(), evcs_dataset_windows.xs(
        3, level=1
    ).copy()

    # # Create features' temporal sequence (i.e. training dataset)
    identity_function = FunctionTransformer(None) # We do not need a scaler for now, so we use the Identity function...
    train_dataset = ds.EDFDataset_v2(building_tokens, model_tokens, building_token_lookup, model_token_lookup, train_delta_windows, scaler=identity_function)
    dev_dataset, test_dataset = ds.EDFDataset_v2(building_tokens, model_tokens, building_token_lookup, model_token_lookup, dev_delta_windows, scaler=train_dataset.scaler),\
                                ds.EDFDataset_v2(building_tokens, model_tokens, building_token_lookup, model_token_lookup, test_delta_windows, scaler=train_dataset.scaler)

    train_loader, dev_loader, test_loader = DataLoader(train_dataset, batch_size=args.bs, shuffle=True, collate_fn=train_dataset.pad_collate),\
                                            DataLoader(dev_dataset,   batch_size=args.bs, shuffle=False, collate_fn=dev_dataset.pad_collate),\
                                            DataLoader(test_dataset,  batch_size=args.bs, shuffle=False, collate_fn=test_dataset.pad_collate)


    # %%
    # # Instantiate ML Model
    device = torch.device(f'cuda:{args.gpuid}') if torch.cuda.is_available() else torch.device('cpu')

    model_params = dict(
        location_embeddings=building_embeddings,
        model_embeddings=model_embeddings,
        input_size=len(windowing_params['rnn_feats']),
        # scale=dict(
        #     sigma=torch.Tensor(train_dataset.scaler.scale_[:1]), 
        #     mu=torch.Tensor(train_dataset.scaler.mean_[:1])
        # ),
        scale=None,
        rnn_cell=getattr(nn, args.rnn_cell.upper()),
        bidirectional=args.bi,
        num_layers=args.num_layers,
        hidden_size=args.hidden_size,
        misc_size=len(windowing_params['fc_feats']),
        fc_layers=[int(i) for i in args.fc_layers.split(',')],
        output_size=len(windowing_params['y_feats']),
    )

    evaluate_fun_params = dict(
        unit='kW',
        metrics_funs=[hl.smape, hl.wape],
        heads=1
    )

    model = ml.EnergyDemandForecasting_v2(**model_params)
    model.to(device)

    print(model)
    print(f'{device=}')
    # pdb.set_trace()


    # %%
    # # Train Model
    criterion = tr.PinballLoss(quantile=0.7)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

    model_name_base = f'{"bi-" if model_params["bidirectional"] else ""}'+\
                      f'{args.rnn_cell}_{model_params["num_layers"]}_'+\
                      f'{model_params["hidden_size"]}_fc_{"_".join(map(str, model_params["fc_layers"]))}_'+\
                      f"{len(windowing_params['rnn_feats'])}_rnn_inputs_{len(windowing_params['fc_feats'])}_fc_inputs_{len(windowing_params['y_feats'])}_outputs_"+\
                      f'batchsize_{args.bs}_patience_{args.patience}__'+\
                      f"{args.data}_dataset_{args.sr_freq}_sequences_"+\
                      f'window_{windowing_params["length_max"]}_stride_{windowing_params["stride"]}.{criterion}.dropout_after_cat.cml.epoch{{0}}.pth'
                      
    os.makedirs(os.path.join('.', 'data', 'pth', f'{model_name_base.split(".")[0]}'), exist_ok=True)
    save_path_epoch = os.path.join('.', 'data', 'pth', f'{model_name_base.split(".")[0]}', model_name_base)
    save_path_best = os.path.join('.', 'data', 'pth', f'{model_name_base.split(".")[0]}', model_name_base.format('best'))
    print(save_path_epoch)

    early_stop_params = dict(
        patience=args.patience,
        save_best=True,
        path=save_path_best,
        min_delta=1e-5
    )
    
    save_current_params = dict(
        path=save_path_epoch
    )

    # %%
    if not args.skip_train:
        tr.train_model(
            model, device, criterion, optimizer, args.n_epochs, 
            train_loader, dev_loader, early_stop=True, save_current=True, 
            evaluate_fun=tr.evaluate_model_multihead_rnn, 
            evaluate_fun_params=evaluate_fun_params,
            early_stop_params=early_stop_params, save_current_params=save_current_params
        )

    # %%
    # # Evaluate Best Model
    checkpoint = torch.load(save_path_best)

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    tr.evaluate_model_multihead_rnn(model, device, criterion, test_loader, desc='ADE @ Test Set...', **evaluate_fun_params)
