# Word BiLSTM one-at-a-time parameter sweep

Offline experiment. Does **not** overwrite `reports/lstm/`.

Same 71,370-row `llm_intent_v1` corpus, same 70/20/10 split (`random_state=42`), same VAL rule (maximize scam recall subject to legitimate recall ≥ 0.85). Every retrain searched VAL thresholds **0.20, 0.25, …, 0.70**.

One-factor-at-a-time first (exactly one training knob vs the published recipe). After those retrains, two or three **combo** runs merge the best distinct groups.

- Catalog OFAT runs: `31`
- Catalog combo runs: `2`
- Finished retrains: `33`

## Best combination (TEST combined mean)

- Run: `07_epochs_8`
- Changed: `epochs = 8`
- Threshold: `0.2`
- TEST scam recall: `0.9913644214162349`
- TEST ham precision: `0.9914627205463858`
- TEST accuracy: `0.9707159871094297`
- Combined mean: `0.9845143763573502`
- TEST missed / ham warned: `30 / 179`
- Chat missed / ham warned: `11 / 29`

Combined mean = equal-weight mean of TEST scam recall, legitimate precision, and accuracy. Chat eval was scored after freeze and was **not** used to pick the ranking.

## Best chat-eval combination (not used for TEST ranking)

- Run: `09_max_tokens_192`
- Chat scam recall: `0.96`
- Chat missed / ham warned: `4 / 34`

## Ranking (TEST combined mean)

| Rank | Run | Changed | Thr | TEST scam recall | TEST ham precision | TEST accuracy | Combined mean | TEST missed / ham warned | Chat missed / ham warned |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `07_epochs_8` | `epochs=8` | `0.2` | 0.9914 | 0.9915 | 0.9707 | 0.9845 | 30 / 179 | 11 / 29 |
| 2 | `03_learning_rate_5e-3` | `learning_rate=0.005` | `0.2` | 0.9911 | 0.9912 | 0.9710 | 0.9844 | 31 / 176 | 12 / 28 |
| 3 | `12_embed_dim_256` | `embed_dim=256` | `0.2` | 0.9908 | 0.9908 | 0.9651 | 0.9822 | 32 / 217 | 6 / 33 |
| 4 | `06_epochs_6` | `epochs=6` | `0.2` | 0.9893 | 0.9894 | 0.9676 | 0.9821 | 37 / 194 | 10 / 37 |
| 5 | `10_max_tokens_256` | `max_tokens=256` | `0.2` | 0.9902 | 0.9902 | 0.9643 | 0.9816 | 34 / 221 | 12 / 41 |
| 6 | `23_batch_size_64` | `batch_size=64` | `0.2` | 0.9859 | 0.9863 | 0.9725 | 0.9816 | 49 / 147 | 9 / 30 |
| 7 | `31_combo_epochs_8__learning_rate_5e-3` | `combo={'epochs': 8, 'learning_rate': 0.005}` | `0.2` | 0.9824 | 0.9832 | 0.9788 | 0.9815 | 61 / 90 | 14 / 20 |
| 8 | `27_grad_clip_0.5` | `grad_clip=0.5` | `0.2` | 0.9856 | 0.9860 | 0.9727 | 0.9814 | 50 / 145 | 11 / 36 |
| 9 | `02_learning_rate_2e-3` | `learning_rate=0.002` | `0.2` | 0.9896 | 0.9897 | 0.9647 | 0.9813 | 36 / 216 | 10 / 31 |
| 10 | `00_baseline_expanded_grid` | `threshold_grid=expanded_0.20_to_0.70_step_0.05` | `0.2` | 0.9842 | 0.9846 | 0.9721 | 0.9803 | 55 / 144 | 12 / 30 |
| 11 | `32_combo_epochs_8__learning_rate_5e-3__embed_dim_256` | `combo={'epochs': 8, 'learning_rate': 0.005, 'embed_dim': 256}` | `0.2` | 0.9807 | 0.9816 | 0.9784 | 0.9802 | 67 / 87 | 14 / 18 |
| 12 | `30_url_features_false` | `url_features=False` | `0.2` | 0.9856 | 0.9859 | 0.9688 | 0.9801 | 50 / 173 | 12 / 33 |
| 13 | `09_max_tokens_192` | `max_tokens=192` | `0.2` | 0.9888 | 0.9888 | 0.9626 | 0.9800 | 39 / 228 | 4 / 34 |
| 14 | `14_hidden_size_256` | `hidden_size=256` | `0.2` | 0.9853 | 0.9856 | 0.9688 | 0.9799 | 51 / 172 | 10 / 42 |
| 15 | `22_max_vocab_size_50000` | `max_vocab_size=50000` | `0.2` | 0.9859 | 0.9861 | 0.9669 | 0.9796 | 49 / 187 | 10 / 45 |
| 16 | `04_epochs_3` | `epochs=3` | `0.2` | 0.9862 | 0.9863 | 0.9658 | 0.9794 | 48 / 196 | 10 / 42 |
| 17 | `20_max_vocab_size_10000` | `max_vocab_size=10000` | `0.2` | 0.9885 | 0.9885 | 0.9609 | 0.9793 | 40 / 239 | 9 / 36 |
| 18 | `18_dropout_0.2` | `dropout=0.2` | `0.2` | 0.9856 | 0.9858 | 0.9662 | 0.9792 | 50 / 191 | 9 / 26 |
| 19 | `published_lstm_original_grid` | `threshold_grid=original_0.30_to_0.70_step_0.05` | `0.3` | 0.9859 | 0.9860 | 0.9651 | 0.9790 | 49 / 200 | 14 / 24 |
| 20 | `21_max_vocab_size_15000` | `max_vocab_size=15000` | `0.2` | 0.9824 | 0.9829 | 0.9702 | 0.9785 | 61 / 152 | 15 / 31 |
| 21 | `08_max_tokens_64` | `max_tokens=64` | `0.2` | 0.9845 | 0.9847 | 0.9654 | 0.9782 | 54 / 193 | 11 / 39 |
| 22 | `05_epochs_5` | `epochs=5` | `0.2` | 0.9954 | 0.9951 | 0.9438 | 0.9781 | 16 / 385 | 10 / 48 |
| 23 | `15_num_layers_2` | `num_layers=2` | `0.2` | 0.9816 | 0.9821 | 0.9700 | 0.9779 | 64 / 150 | 12 / 25 |
| 24 | `29_class_weight_none` | `class_weight=none` | `0.2` | 0.9845 | 0.9846 | 0.9644 | 0.9778 | 54 / 200 | 13 / 31 |
| 25 | `17_dropout_0` | `dropout=0.0` | `0.2` | 0.9796 | 0.9803 | 0.9730 | 0.9776 | 71 / 122 | 16 / 22 |
| 26 | `11_embed_dim_64` | `embed_dim=64` | `0.2` | 0.9830 | 0.9833 | 0.9658 | 0.9774 | 59 / 185 | 15 / 39 |
| 27 | `01_learning_rate_5e-4` | `learning_rate=0.0005` | `0.2` | 0.9902 | 0.9899 | 0.9484 | 0.9762 | 34 / 334 | 11 / 46 |
| 28 | `24_batch_size_256` | `batch_size=256` | `0.2` | 0.9937 | 0.9933 | 0.9392 | 0.9754 | 22 / 412 | 10 / 54 |
| 29 | `16_num_layers_3` | `num_layers=3` | `0.2` | 0.9750 | 0.9761 | 0.9728 | 0.9746 | 87 / 107 | 15 / 21 |
| 30 | `13_hidden_size_64` | `hidden_size=64` | `0.2` | 0.9787 | 0.9791 | 0.9633 | 0.9737 | 74 / 188 | 15 / 38 |
| 31 | `28_grad_clip_2` | `grad_clip=2.0` | `0.2` | 0.9738 | 0.9749 | 0.9700 | 0.9729 | 91 / 123 | 14 / 33 |
| 32 | `19_dropout_0.5` | `dropout=0.5` | `0.2` | 0.9899 | 0.9894 | 0.9392 | 0.9728 | 35 / 399 | 5 / 45 |
| 33 | `25_weight_decay_1e-4` | `weight_decay=0.0001` | `0.2` | 0.9712 | 0.9722 | 0.9622 | 0.9685 | 100 / 170 | 10 / 47 |
| 34 | `26_weight_decay_1e-3` | `weight_decay=0.001` | `0.2` | 0.9637 | 0.9638 | 0.9395 | 0.9557 | 126 / 306 | 11 / 57 |
