# Ensemble Analysis

### Requirements

Paciencia++

Pytoch >= 1.6 


# Procedure

For ensembling first generate the logits with evaluate. Next, move all the desired logits to a directory 
and pass it through argument to `ensemble.py`.

### ToDo

- SWAG

## Results

### CIFAR-10

| Model |      Info      | Validation |  Test  |
|:-----:|:--------------:|:----------:|:------:|
| Res18 | Seed 311220201 |   0.9431   | 0.9473 |
| Res18 | Seed 311220202 |   0.9441   | 0.9437 |
| Res18 | Seed 311220203 |   0.9482   | 0.9467 |
| Res18 | Seed 311220204 |   0.9425   | 0.9447 |
| Res18 | Seed 311220205 |   0.9414   | 0.9457 |

#### Raw

---- Validation evaluation ----


|                           Method                           | Accuracy |  ECE   |  MCE   | BRIER  |  NNL   |
|:----------------------------------------------------------:|:--------:|:------:|:------:|:------:|:------:|
| model1/val_logits_model_kuangliu_resnet18_best_accuracy.pt |  0.9478  | 2.4941 | 1.2195 | 0.0081 | 0.1915 |
| model2/val_logits_model_kuangliu_resnet18_best_accuracy.pt |  0.9488  | 2.6258 | 1.5179 | 0.0083 | 0.1925 |
| model3/val_logits_model_kuangliu_resnet18_best_accuracy.pt |   0.95   | 2.6483 | 1.3220 | 0.0081 | 0.1883 |
| model4/val_logits_model_kuangliu_resnet18_best_accuracy.pt |  0.9472  | 2.9554 | 1.5557 | 0.0086 | 0.2001 |
| model5/val_logits_model_kuangliu_resnet18_best_accuracy.pt |  0.949   | 2.6576 | 1.4270 | 0.0083 | 0.1937 |
|                     Avg probs ensemble                     |  0.9582  | 0.8310 | 0.3613 | 0.0065 | 0.1394 |
_____________________________________________________________________________________________________________

---- Test evaluation ----


|                           Method                            | Accuracy |  ECE   |  MCE   | BRIER  |  NNL   |
|:-----------------------------------------------------------:|:--------:|:------:|:------:|:------:|:------:|
| model1/test_logits_model_kuangliu_resnet18_best_accuracy.pt |  0.9482  | 2.4885 | 1.3318 | 0.0082 | 0.1930 |
| model2/test_logits_model_kuangliu_resnet18_best_accuracy.pt |  0.9438  | 2.9105 | 1.5244 | 0.0090 | 0.2085 |
| model3/test_logits_model_kuangliu_resnet18_best_accuracy.pt |  0.9469  | 2.5997 | 1.3823 | 0.0084 | 0.1900 |
| model4/test_logits_model_kuangliu_resnet18_best_accuracy.pt |  0.9448  | 2.7649 | 1.4504 | 0.0088 | 0.2045 |
| model5/test_logits_model_kuangliu_resnet18_best_accuracy.pt |  0.9458  | 2.6418 | 1.4205 | 0.0085 | 0.1966 |
|                     Avg probs ensemble                      |  0.9562  | 0.7271 | 0.3390 | 0.0067 | 0.1424 |
______________________________________________________________________________________________________________

#### Temperature Scaling

---- Validation evaluation ----

|                           Method                           | Accuracy |  ECE   |  MCE   | BRIER  |  NNL   |
|:----------------------------------------------------------:|:--------:|:------:|:------:|:------:|:------:|
| model1/val_logits_model_kuangliu_resnet18_best_accuracy.pt |  0.9478  | 0.9427 | 0.1853 | 0.0078 | 0.1756 |
| model2/val_logits_model_kuangliu_resnet18_best_accuracy.pt |  0.9488  | 0.9098 | 0.2069 | 0.0080 | 0.1724 |
| model3/val_logits_model_kuangliu_resnet18_best_accuracy.pt |   0.95   | 1.0319 | 0.2428 | 0.0077 | 0.1720 |
| model4/val_logits_model_kuangliu_resnet18_best_accuracy.pt |  0.9472  | 1.0328 | 0.2270 | 0.0081 | 0.1787 |
| model5/val_logits_model_kuangliu_resnet18_best_accuracy.pt |  0.949   | 0.9164 | 0.2222 | 0.0079 | 0.1770 |
|                     Avg probs ensemble                     |  0.9586  | 1.5347 | 0.6025 | 0.0065 | 0.1425 |

---- Test evaluation ----

|                           Method                            | Accuracy |  ECE   |  MCE   | BRIER  |  NNL   |
|:-----------------------------------------------------------:|:--------:|:------:|:------:|:------:|:------:|
| model1/test_logits_model_kuangliu_resnet18_best_accuracy.pt |  0.9482  | 0.8139 | 0.1761 | 0.0079 | 0.1774 |
| model2/test_logits_model_kuangliu_resnet18_best_accuracy.pt |  0.9438  | 0.7971 | 0.2085 | 0.0086 | 0.1858 |
| model3/test_logits_model_kuangliu_resnet18_best_accuracy.pt |  0.9469  | 0.8008 | 0.1965 | 0.0080 | 0.1754 |
| model4/test_logits_model_kuangliu_resnet18_best_accuracy.pt |  0.9448  | 0.7183 | 0.3047 | 0.0084 | 0.1843 |
| model5/test_logits_model_kuangliu_resnet18_best_accuracy.pt |  0.9458  | 0.7023 | 0.1694 | 0.0082 | 0.1821 |
|                     Avg probs ensemble                      |  0.9562  | 1.6108 | 0.6237 | 0.0067 | 0.1471 |



### Benchmarks

https://benchmarks.ai/cifar-10

