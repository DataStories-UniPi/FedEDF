import pandas as pd
import numpy as np

import helper as hl
import pmdarima as pmd


def stat_model_fit(X_train, time_name='timestamp', oid_name='oid', target_name='power_curr', model=pmd.arima.ARIMA, model_name=None, model_params=None, n_jobs=-1, **kwargs):
    X_feats = kwargs.pop('X_feats', None)

    stat_models = hl.applyParallel(
        X_train.sort_values(time_name).groupby(oid_name),
        lambda l: pd.Series(
            [
                model(
                    **model_params
                ).fit(
                    l[[target_name]].values, 
                    l[X_feats].values if X_feats else None
                )
            ],
            index=[model_name]
        ) if len(l) > 1 else None,
        n_jobs=n_jobs,
    )
    return stat_models


def stat_model_predict(X_test, stat_models, model_name, time_name='timestamp', oid_name='oid', **kwargs):
    X_feats = kwargs.pop('X_feats', None)

    y_pred = X_test.loc[
        # Get predictions only for eligible EVSEs (i.e., have a trained model)
        X_test[oid_name].isin(stat_models.index.get_level_values(0))
    ].sort_values(time_name).groupby(oid_name).apply(
        lambda l: pd.Series(
            data = stat_models.loc[l.name, model_name].predict(
                len(l),
                l[X_feats].values if X_feats else None
            ),
            index = l.timestamp
        )
    )
    return y_pred
