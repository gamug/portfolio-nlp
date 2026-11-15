# The arms on the old tags: 1222 rows, 1048 articles, 1000 rows with one tag

| arm | file | top = first tag | top is any tag | single-tag rows: accuracy | macro F1 |
|---|---|---|---|---|---|
| untouched zero-shot | `oof_base.npz` | 59.7% | 66.0% | 65.0% | 0.429 |
| entailment fine-tune, every tag a positive | `oof_multi.npz` | 79.4% | 82.7% | 82.2% | 0.654 |
| head on the zero-shot encoder (12 sigmoid outputs) | `oof_head.npz` | 65.0% | 66.6% | 65.2% | 0.267 |
| plain DeBERTa-v3-base, 12 sigmoid outputs | `oof_deb.npz` | 72.5% | 77.5% | 76.8% | 0.585 |
