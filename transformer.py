import math
import polars as pl
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

####################
MAX_LEN = 32
BATCH_SIZE = 256
NUM_EPOCHS = 12
LR = 3e-4
WEIGHT_DECAY = 1e-2
WARMUP_RATIO = 0.08
DEVICE = "cuda"

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


class CreditDataset(Dataset):

    def __init__(self, df, has_label = True):
        df = df.sort(['id', 'rn'])
        ids_arr = df['id'].to_numpy()

        # Fast dataset construction
        change_points = np.where(ids_arr[:-1] != ids_arr[1:])[0] + 1
        self.starts = np.concatenate([[0], change_points])
        self.ends = np.concatenate([change_points, [len(ids_arr)]])
        self.unique_ids = ids_arr[self.starts].astype(np.int64)

        self.features = np.stack(
            [df[col].to_numpy() for col in FEATURE_COLS], axis=1
        ).astype(np.int16)  # to save memory

        self.labels = (
            df['flag'].to_numpy()[self.starts].astype(np.float32)
            if has_label
            else None
        )

    def __len__(self):
        return len(self.unique_ids)

    def __getitem__(self, idx):
        start, end = self.starts[idx], self.ends[idx]
        return {
            'features': self.features[start:end],
            'id': self.unique_ids[idx],
            'label': None if self.labels is None else self.labels[idx],
        }


def collate(batch):
    B = len(batch)
    seq_lens = np.fromiter(
        [min(item['features'].shape[0], MAX_LEN) for item in batch],
        dtype=np.int64, count=B,
    )
    L = int(seq_lens.max())
    n_features = batch[0]['features'].shape[1]

    out = np.zeros((B, L, n_features), dtype=np.int64)
    for i, item in enumerate(batch):
        k = seq_lens[i]
        # Apply reverse order for embeddings
        out[i, :k] = item['features'][-k:][::-1].astype(np.int64) + 1

    pad_mask = torch.from_numpy(np.arange(L)[None, :] >= seq_lens[:, None])

    feature_tensor = torch.from_numpy(out)  # (B, L, n_features)
    features = {col: feature_tensor[..., i] for i, col in enumerate(FEATURE_COLS)}
    ids = torch.from_numpy(np.array([item['id'] for item in batch], dtype=np.int64))

    if batch[0]['label'] is not None:
        labels = torch.from_numpy(np.array([item['label'] for item in batch], dtype=np.float32))
        return features, pad_mask, labels, ids

    return features, pad_mask, None, ids


def compute_vocab_sizes(dfs, feature_cols):
    sizes = {}
    for col in feature_cols:
        # values are in {0, ... , m} -> max + 1 slots
        m = max(int(df[col].max()) for df in dfs)
        # +1 for padding
        sizes[col] = m + 2
    return sizes


class CreditTransformer(nn.Module):

    def __init__(
        self,
        vocab_sizes,
        d_model=128,
        nhead=8,
        num_layers=3,
        max_len=MAX_LEN,
        dropout=0.05,
        emb_dim=16,
    ):
        super().__init__()
        
        # embed each feature, so we have len(vocav_sizes) vectors of size emb_dim 
        self.embeddings = nn.ModuleDict({
            name: nn.Embedding(vocab_size, emb_dim, padding_idx=0)
            for name, vocab_size in vocab_sizes.items()
        })
        self.feature_proj = nn.Linear(emb_dim * len(vocab_sizes), d_model)

        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.pos_emb = nn.Embedding(max_len + 1, d_model)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers)

        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, features, pad_mask):
        B, L = pad_mask.shape
        embs = [self.embeddings[name](features[name]) for name in self.embeddings]

        x = self.feature_proj(torch.cat(embs, dim=-1))
        cls = self.cls_token.expand(B, -1, -1)

        # append cls token for classification
        x = torch.cat([cls, x], dim=1)

        positions = torch.arange(L + 1, device=x.device)
        x = x + self.pos_emb(positions)[None]

        cls_mask = torch.zeros(B, 1, dtype=torch.bool, device=x.device)
        full_mask = torch.cat([cls_mask, pad_mask], dim=1)

        x = self.encoder(x, src_key_padding_mask=full_mask)
        return self.head(x[:, 0]).squeeze(-1)


def cosine_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    def lr_lambda(step):
        if step < num_warmup_steps:
            return step / max(1, num_warmup_steps)
        progress = (step - num_warmup_steps) / max(1, num_training_steps - num_warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return LambdaLR(optimizer, lr_lambda)


@torch.no_grad()
def evaluate(model, loader, criterion):
    losses = []
    preds = []
    labels = []

    model.eval()
    for features, mask, y, _ in loader:
        features = {k: v.to(DEVICE) for k, v in features.items()}
        mask, y = mask.to(DEVICE), y.to(DEVICE)

        logits = model(features, mask)

        losses.append(criterion(logits, y).item())
        preds.append(torch.sigmoid(logits).cpu().numpy())
        labels.append(y.cpu().numpy())

    return np.mean(losses), roc_auc_score(np.concatenate(labels), np.concatenate(preds))


@torch.no_grad()
def predict(model, loader):
    all_ids = []
    all_preds = []

    model.eval()
    for features, mask, _, ids in loader:
        features = {k: v.to(DEVICE) for k, v in features.items()}
        mask = mask.to(DEVICE)

        logits = model(features, mask)

        all_preds.append(torch.sigmoid(logits).cpu().numpy())
        all_ids.append(ids.numpy())

    return np.concatenate(all_ids), np.concatenate(all_preds)


def main():
    train_df = pl.read_parquet('data/train_data.parquet')
    test_df = pl.read_parquet('data/test_data.parquet')

    target = pl.read_csv('data/train_target.csv')
    train_df = train_df.join(target, on='id', how='inner')

    VOCAB_SIZES = compute_vocab_sizes([train_df, test_df], FEATURE_COLS)

    ids_sorted = np.sort(train_df['id'].unique().to_numpy())
    train_size = int(len(ids_sorted) * 0.8)
    val_ids = set(ids_sorted[train_size:].tolist())
    is_val = train_df['id'].is_in(val_ids)

    print("Processing data...")
    train_ds = CreditDataset(train_df.filter(~is_val), has_label=True)
    val_ds = CreditDataset(train_df.filter(is_val), has_label=True)
    test_ds = CreditDataset(test_df, has_label=False)

    test_preds = []
    val_preds = []
    ids = None

    do_train = True
    do_predict = True

    for seed in [1, 13, 42, 69, 78]:
        torch.manual_seed(seed)
        np.random.seed(seed)

        train_loader = DataLoader(
            train_ds, batch_size=BATCH_SIZE, shuffle=True,
            collate_fn=collate, num_workers=2, pin_memory=True,
        )
        val_loader = DataLoader(
            val_ds, batch_size=BATCH_SIZE, shuffle=False,
            collate_fn=collate, num_workers=2, pin_memory=True,
        )
        test_loader = DataLoader(
            test_ds, batch_size=BATCH_SIZE, shuffle=False,
            collate_fn=collate, num_workers=2, pin_memory=True,
        )

        model = CreditTransformer(VOCAB_SIZES).to(DEVICE)
        
        if do_train:
            criterion = nn.BCEWithLogitsLoss()
            optim = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

            total_steps = NUM_EPOCHS * len(train_loader)
            warmup_steps = int(WARMUP_RATIO * total_steps)
            scheduler = cosine_with_warmup(optim, warmup_steps, total_steps)

            print(f"Starting training! (with {seed = })")

            best_auc, best_state = 0.0, None
            for epoch in range(NUM_EPOCHS):
                model.train()
                running = 0.0

                for features, mask, y, _ in train_loader:
                    features = {k: v.to(DEVICE) for k, v in features.items()}
                    mask, y = mask.to(DEVICE), y.to(DEVICE)

                    logits = model(features, mask)

                    loss = criterion(logits, y)
                    running += loss.item()

                    optim.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                    optim.step()
                    scheduler.step()

                train_loss = running / len(train_loader)
                val_loss, val_auc = evaluate(model, val_loader, criterion)
                
                print(
                    f'Epoch {epoch + 1:2d}/{NUM_EPOCHS}: train_loss = {train_loss:.4f}, '
                    f'val_loss = {val_loss:.4f}, val_auc = {val_auc:.4f}'
                )
                if val_auc > best_auc:
                    best_auc = val_auc
                    best_state = {
                        k: v.detach().cpu().clone()
                        for k, v in model.state_dict().items()
                    }

            print(f'Best val ROC-AUC: {best_auc:.4f}')
            model.load_state_dict(best_state)
            torch.save(best_state, f'checkpoints/credit_transformer_m{seed}.pt')
        else:
            model.load_state_dict(torch.load(f'checkpoints/credit_transformer_m{seed}.pt'))

        _, val_pred = predict(model, val_loader)
        ids, test_pred = predict(model, test_loader)
        val_preds.append(val_pred)
        test_preds.append(test_pred)

    val_final = np.mean([pd.Series(p).rank(pct=True).values for p in val_preds], axis=0)
    test_final = np.mean([pd.Series(p).rank(pct=True).values for p in test_preds], axis=0)

    labels = []
    for features, mask, y, _ in val_loader:
        labels.append(y.cpu().numpy())
    val_labels = np.concatenate(labels)

    print(f'Multi-seed val ROC-AUC: {roc_auc_score(val_labels, val_final):.5f}')

    if do_predict:
        sub = pd.DataFrame({'id': ids, 'flag': test_final}).sort_values('id')
        sub['flag'] = sub['flag'].round(10)
        sub.to_csv('submissions/transformer_megamix.csv', index=False)

if __name__ == "__main__":
    main()
