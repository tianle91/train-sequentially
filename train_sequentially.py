from typing import List, Dict

import lightgbm as lgb


def lgb_train_sequentially(
        feature_names_sequence: List[List[str]],
        lgbdatasetparams: Dict,
        lgbtrainparams: Dict
):
    '''
    feature_names_sequence: list of list of strings
    lgbdatasetparams: to pass into lightgbm.Dataset for train_set
    lgbtrainparams: to pass into lightgbm.train
    '''
    assert 'train_set' not in lgbtrainparams, 'provide data in lgbdatasetparams'
    assert 'categorical_feature' not in lgbdatasetparams, 'use categorical in pandas'

    data = lgbdatasetparams['data']
    model = None
    feature_names_cumulative = []

    for feature_names in feature_names_sequence:
        assert isinstance(feature_names, list), 'feature_names_sequence is a list of lists'
        assert set(feature_names).issubset(set(data.columns))

        # update feature_names_cumulative
        feature_names_cumulative += feature_names
        feature_names_cumulative = list(set(feature_names_cumulative))

        # update lgbdatasetparams
        lgbdatasetparams_new = lgbdatasetparams.copy()
        lgbdatasetparams_new.update({'data': data[feature_names_cumulative]})
        train_set = lgb.Dataset(**lgbdatasetparams_new)

        # update lgbtrainparams
        lgbtrainparams_new = lgbtrainparams.copy()
        lgbtrainparams_new.update({'init_model': model, 'train_set': train_set})

        model = lgb.train(**lgbtrainparams_new)

    return model


if __name__ == '__main__':
    from sklearn.datasets import load_iris
    import pandas as pd
    import numpy as np

    df = load_iris()
    xcolsseq = df['feature_names']
    X, y = pd.DataFrame(df['data'], columns=df['feature_names']), df['target']

    feature_names_sequence = [[c] for c in xcolsseq]

    model = lgb_train_sequentially(
        feature_names_sequence=feature_names_sequence,
        lgbdatasetparams={
            'data': X,
            'label': y,
        },
        lgbtrainparams={'params': {
                'objective': 'multiclass',
                'num_class': len(np.unique(y))
        }}
    )

    feature_importance = pd.DataFrame({
        'feature_name': model.feature_name(),
        'feature_importance': model.feature_importance(),
    })


    def clean_string(s):
        return ''.join(['_' if c == ' ' else c for c in s])


    train_seq_i = []
    train_seq_c = []
    for i, cols in zip(range(len(feature_names_sequence)), feature_names_sequence):
        for c in cols:
            train_seq_i.append(i)
            train_seq_c.append(clean_string(c))

    train_sequence = pd.DataFrame({
        'train_sequence': train_seq_i,
        'feature_name': train_seq_c,
    })

    print(feature_importance.merge(
        train_sequence,
        on='feature_name',
        how='left'
    ).sort_values('train_sequence'))
