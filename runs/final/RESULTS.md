# MRLHGNN (corrected): 5 seeds x 5-fold CV

seeds: [0, 1, 2, 3, 4]

config: `{"folds": 5, "max_folds": null, "lr": 0.001, "hidden": 128, "dropout": 0.4, "topk": 15, "weight_decay": 0.0, "epochs": 2500, "val_frac": 0.1, "eval_every": 10, "patience": 200, "pos_weight": 50.0, "no_layer_norm": false, "no_transformer": false, "no_decoder_attention": false, "device": "cuda"}`

| metric | mean | std | ensemble of 5 seeds |
|---|---|---|---|
| AUC | 0.8071 | 0.0077 | 0.8460 |
| AUPR | 0.3496 | 0.0096 | 0.4262 |
| F1 | 0.4220 | 0.0063 | 0.4740 |
| Accuracy | 0.9932 | 0.0003 | 0.9939 |
| Recall | 0.3702 | 0.0194 | 0.4149 |
| Specificity | 0.9974 | 0.0004 | 0.9977 |
| Precision | 0.4939 | 0.0289 | 0.5527 |
| sk_AUC | 0.8071 | 0.0077 | 0.8460 |
| sk_AP | 0.3359 | 0.0106 | 0.4265 |

| seed | AUC | AUPR | F1 | Accuracy | Recall | Specificity | Precision | sk_AUC | sk_AP |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.8081 | 0.3630 | 0.4309 | 0.9930 | 0.3979 | 0.9970 | 0.4699 | 0.8082 | 0.3487 |
| 1 | 0.8169 | 0.3513 | 0.4243 | 0.9931 | 0.3828 | 0.9972 | 0.4759 | 0.8169 | 0.3387 |
| 2 | 0.7940 | 0.3434 | 0.4130 | 0.9930 | 0.3709 | 0.9971 | 0.4659 | 0.7941 | 0.3275 |
| 3 | 0.8118 | 0.3551 | 0.4250 | 0.9936 | 0.3569 | 0.9978 | 0.5253 | 0.8119 | 0.3446 |
| 4 | 0.8045 | 0.3351 | 0.4168 | 0.9936 | 0.3425 | 0.9980 | 0.5325 | 0.8045 | 0.3202 |
