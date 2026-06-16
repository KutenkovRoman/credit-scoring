import polars as pl
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

from catboost import CatBoostClassifier

import torch
import torch.nn as nn

from transformer import (
    DEVICE, BATCH_SIZE, FEATURE_COLS,
    CreditDataset,
    collate, compute_vocab_sizes,
    CreditTransformer,
    predict,
)
from catbooster import make_features


def rank_blend(preds_list, weights):
    weights = np.array(weights) / sum(weights)
    blended = np.zeros_like(preds_list[0], dtype=np.float64)

    for preds, w in zip(preds_list, weights):
        # rank(pct=True) -> (0, 1]
        blended += w * pd.Series(preds).rank(pct=True).values

    return blended


def main():
    train_df = pl.read_parquet('data/train_data.parquet')
    test_df = pl.read_parquet('data/test_data.parquet')

    print('Building features...')
    X_train = make_features(train_df)
    X_test = make_features(test_df)

    target = pl.read_csv('data/train_target.csv')
    X_train = X_train.join(target, on='id', how='inner')
    train_df = train_df.join(target, on='id', how='inner')

    ids_sorted = np.sort(train_df['id'].unique().to_numpy())
    train_size = int(len(ids_sorted) * 0.8)
    val_ids = set(ids_sorted[train_size:].tolist())

    # Catboost preds
    catboost_is_val = X_train['id'].is_in(val_ids)
    X_val_interm = X_train.filter(catboost_is_val)

    drop_cols = ['id', 'flag']
    y_val = X_val_interm['flag'].to_numpy()
    X_val = X_val_interm.drop(drop_cols).to_pandas()

    test_ids = X_test['id'].to_numpy()
    X_test = X_test.drop('id').to_pandas()

    catbooster = CatBoostClassifier()
    catbooster.load_model('checkpoints/catbooster_v2.cbm')

    catboost_val_preds = catbooster.predict_proba(X_val)[:, 1]
    catboost_test_preds = catbooster.predict_proba(X_test)[:, 1]

    # Transformer preds
    trans_is_val = train_df['id'].is_in(val_ids)

    print('Processing data...')
    val_ds = CreditDataset(train_df.filter(trans_is_val), has_label=True)
    test_ds = CreditDataset(test_df, has_label=False)

    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=collate, num_workers=2, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=collate, num_workers=2, pin_memory=True,
    )

    VOCAB_SIZES = compute_vocab_sizes([train_df, test_df], FEATURE_COLS)

    test_preds = []
    val_preds = []
    test_ids = None

    for seed in [1, 13, 42, 69, 78]:
        transformer = CreditTransformer(VOCAB_SIZES).to(DEVICE)
        transformer.load_state_dict(torch.load(f'checkpoints/credit_transformer_m{seed}.pt'))

        _, val_pred = predict(transformer, val_loader)
        test_ids, test_pred = predict(transformer, test_loader)

        val_preds.append(val_pred)
        test_preds.append(test_pred)

    trans_val_preds = np.mean([pd.Series(p).rank(pct=True).values for p in val_preds], axis=0)
    trans_test_preds = np.mean([pd.Series(p).rank(pct=True).values for p in test_preds], axis=0)

    # Weight tuning
    print(f'CatBoost AUC: {roc_auc_score(y_val, catboost_val_preds):.5f}')
    print(f'Transformer AUC: {roc_auc_score(y_val, trans_val_preds):.5f}')

    best_auc = -1
    best_w = 0

    for w in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        blend = rank_blend([catboost_val_preds, trans_val_preds], [1 - w, w])
        auc = roc_auc_score(y_val, blend)

        if auc > best_auc:
            best_auc = auc
            best_w = w

        print(f'Blend (w_catboost = {1 - w:.1f}, w_transformer = {w:.1f}) AUC: {auc:.5f}')

    test_blend = rank_blend([catboost_test_preds, trans_test_preds], [1 - best_w, best_w])

    submission = pd.DataFrame({'id': test_ids, 'flag': test_blend.round(10)})
    submission.sort_values('id').to_csv('submissions/blend.csv', index=False)


if __name__ == "__main__":
    main()
