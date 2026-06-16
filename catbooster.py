import polars as pl
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score

####################
ITERATIONS = 5000
LR = 0.03
DEPTH = 7
DEVICE = "GPU"

FEATURE_COLS = [
    "pre_since_opened",
    "pre_since_confirmed",
    "pre_pterm",
    "pre_fterm",
    "pre_till_pclose",
    "pre_till_fclose",
    "pre_loans_credit_limit",
    "pre_loans_next_pay_summ",
    "pre_loans_outstanding",
    "pre_loans_total_overdue",
    "pre_loans_max_overdue_sum",
    "pre_loans_credit_cost_rate",
    "pre_loans5",
    "pre_loans530",
    "pre_loans3060",
    "pre_loans6090",
    "pre_loans90",
    "is_zero_loans5",
    "is_zero_loans530",
    "is_zero_loans3060",
    "is_zero_loans6090",
    "is_zero_loans90",
    "pre_util",
    "pre_over2limit",
    "pre_maxover2limit",
    "is_zero_util",
    "is_zero_over2limit",
    "is_zero_maxover2limit",
    "enc_paym_0",
    "enc_paym_1",
    "enc_paym_2",
    "enc_paym_3",
    "enc_paym_4",
    "enc_paym_5",
    "enc_paym_6",
    "enc_paym_7",
    "enc_paym_8",
    "enc_paym_9",
    "enc_paym_10",
    "enc_paym_11",
    "enc_paym_12",
    "enc_paym_13",
    "enc_paym_14",
    "enc_paym_15",
    "enc_paym_16",
    "enc_paym_17",
    "enc_paym_18",
    "enc_paym_19",
    "enc_paym_20",
    "enc_paym_21",
    "enc_paym_22",
    "enc_paym_23",
    "enc_paym_24",
    "enc_loans_account_holder_type",
    "enc_loans_credit_status",
    "enc_loans_credit_type",
    "enc_loans_account_cur",
    "pclose_flag",
    "fclose_flag",
]
####################


def make_features(df):
    df = df.sort(['id', 'rn'])
    exprs = [
        pl.len().alias('n_credits'),
        pl.col('rn').max().alias('max_rn'),
    ]
    for c in FEATURE_COLS:
        exprs += [
            pl.col(c).mean().alias(f'{c}__mean'),
            pl.col(c).max().alias(f'{c}__max'),
            pl.col(c).min().alias(f'{c}__min'),
            pl.col(c).std().alias(f'{c}__std'),
            pl.col(c).n_unique().alias(f'{c}__nunique'),
            pl.col(c).first().alias(f'{c}__first'),  # oldest
            pl.col(c).last().alias(f'{c}__last'),    # newest
        ]

    return df.group_by('id').agg(exprs).sort('id')


def main():
    train = pl.read_parquet('data/train_data.parquet')
    test = pl.read_parquet('data/test_data.parquet')
    target = pl.read_csv('data/train_target.csv')

    print('Building features...')
    X_train = make_features(train)
    X_test = make_features(test)

    X_train = X_train.join(target, on='id', how='inner')

    ids_sorted = np.sort(X_train['id'].to_numpy())
    train_size = int(len(ids_sorted) * 0.8)
    val_ids = set(ids_sorted[train_size:].tolist())

    is_val = X_train['id'].is_in(val_ids)
    X_train_interm = X_train.filter(~is_val)
    X_val_interm = X_train.filter(is_val)

    drop_cols = ['id', 'flag']
    y_train = X_train_interm['flag'].to_numpy()
    y_val = X_val_interm['flag'].to_numpy()
    X_train = X_train_interm.drop(drop_cols).to_pandas()
    X_val = X_val_interm.drop(drop_cols).to_pandas()
    X_test = X_test.drop('id').to_pandas()
    test_ids = X_test['id'].to_numpy()

    cat_features = [
        c for c in X_train.columns
        if (
            (c.startswith(('enc_loans_', 'enc_paym_'))) and
            (c.endswith('__last') or c.endswith('__first'))
        )
    ]
    for c in cat_features:
        X_train[c] = X_train[c].fillna(-1).astype(int)
        X_val[c] = X_val[c].fillna(-1).astype(int)
        X_test[c] = X_test[c].fillna(-1).astype(int)

    print('Fitting model...')
    model = CatBoostClassifier(
        iterations=ITERATIONS,
        learning_rate=LR,
        depth=DEPTH,
        l2_leaf_reg=5.0,
        eval_metric='AUC',
        early_stopping_rounds=300,
        cat_features=cat_features,
        random_seed=42,
        verbose=200,
        thread_count=-1,
        task_type=device,
        devices='0',
    )
    model.fit(X_train, y_train, eval_set=(X_val, y_val))

    val_pred = model.predict_proba(X_val)[:, 1]
    print(f'Validation ROC-AUC: {roc_auc_score(y_val, val_pred):.4f}')

    model.save_model('checkpoints/catbooster_v2.cbm')

    test_pred = model.predict_proba(X_test)[:, 1]
    pd.DataFrame(
        {'id': test_ids, 'flag': test_pred}
    ).to_csv('submissions/catboost_v2.csv', index=False)


if __name__ == "__main__":
    main()
