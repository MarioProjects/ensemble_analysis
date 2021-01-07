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

#### Validation

|              Method             | Accuracy | T.Accuracy |   ECE  |  T.ECE |   MCE  |  T.MCE |  BRIER | T.BRIER |   NNL  |  T.NNL |
|:-------------------------------:|:--------:|:----------:|:------:|:------:|:------:|:------:|:------:|:-------:|:------:|:------:|
| kuangliu_resnet18_best_accuracy |  0.9478  |   0.9478   | 2.4941 | 0.9427 | 1.2195 | 0.1853 | 0.0081 |  0.0078 | 0.1915 | 0.1756 |
| kuangliu_resnet18_best_accuracy |  0.9488  |   0.9488   | 2.6258 | 0.9098 | 1.5179 | 0.2069 | 0.0083 |  0.0080 | 0.1925 | 0.1724 |
| kuangliu_resnet18_best_accuracy |  0.9500  |   0.9500   | 2.6483 | 1.0319 | 1.3220 | 0.2428 | 0.0081 |  0.0077 | 0.1883 | 0.1720 |
| kuangliu_resnet18_best_accuracy |  0.9472  |   0.9472   | 2.9554 | 1.0328 | 1.5557 | 0.2270 | 0.0086 |  0.0081 | 0.2001 | 0.1787 |
| kuangliu_resnet18_best_accuracy |  0.9490  |   0.9490   | 2.6576 | 0.9164 | 1.4270 | 0.2222 | 0.0083 |  0.0079 | 0.1937 | 0.1770 |
|        Avg probs ensemble       |  0.9582  |   0.9586   | 0.8310 | 1.5347 | 0.3613 | 0.6025 | 0.0065 |  0.0065 | 0.1394 | 0.1425 |


|              Method             | Accuracy | T.Accuracy |   ECE  |  T.ECE |   MCE  |  T.MCE |  BRIER | T.BRIER |   NNL  |  T.NNL |
|:-------------------------------:|:--------:|:----------:|:------:|:------:|:------:|:------:|:------:|:-------:|:------:|:------:|
| kuangliu_densenet_best_accuracy |  0.9478  |   0.9478   | 2.7733 | 1.0289 | 1.6295 | 0.2698 | 0.0087 |  0.0083 | 0.2189 | 0.1859 |
| kuangliu_densenet_best_accuracy |   0.944  |   0.9440   | 3.1934 | 1.2235 | 1.6521 | 0.2849 | 0.0090 |  0.0084 | 0.2174 | 0.1860 |
| kuangliu_densenet_best_accuracy |  0.9472  |   0.9472   | 3.0256 | 1.2126 | 1.7703 | 0.2438 | 0.0089 |  0.0085 | 0.2098 | 0.1800 |
| kuangliu_densenet_best_accuracy |  0.9488  |   0.9488   | 3.0765 | 0.7734 | 1.8435 | 0.1538 | 0.0086 |  0.0081 | 0.2072 | 0.1769 |
| kuangliu_densenet_best_accuracy |   0.946  |   0.9460   | 3.1373 | 1.2684 | 1.8365 | 0.5098 | 0.0089 |  0.0084 | 0.2213 | 0.1862 |
|        Avg probs ensemble       |  0.9588  |   0.9582   | 1.0417 | 2.3740 | 0.2918 | 0.9493 | 0.0064 |  0.0065 | 0.1372 | 0.1433 |



#### Test

|              Method             | Accuracy | T.Accuracy |   ECE  |  T.ECE |   MCE  |  T.MCE |  BRIER | T.BRIER |   NNL  |  T.NNL |
|:-------------------------------:|:--------:|:----------:|:------:|:------:|:------:|:------:|:------:|:-------:|:------:|:------:|
| kuangliu_resnet18_best_accuracy |  0.9482  |   0.9482   | 2.4885 | 0.8139 | 1.3318 | 0.1761 | 0.0082 |  0.0079 | 0.1930 | 0.1774 |
| kuangliu_resnet18_best_accuracy |  0.9438  |   0.9438   | 2.9105 | 0.7971 | 1.5244 | 0.2085 | 0.0090 |  0.0086 | 0.2085 | 0.1858 |
| kuangliu_resnet18_best_accuracy |  0.9469  |   0.9469   | 2.5997 | 0.8008 | 1.3823 | 0.1965 | 0.0084 |  0.0080 | 0.1900 | 0.1754 |
| kuangliu_resnet18_best_accuracy |  0.9448  |   0.9448   | 2.7649 | 0.7183 | 1.4504 | 0.3047 | 0.0088 |  0.0084 | 0.2045 | 0.1843 |
| kuangliu_resnet18_best_accuracy |  0.9458  |   0.9458   | 2.6418 | 0.7023 | 1.4205 | 0.1694 | 0.0085 |  0.0082 | 0.1966 | 0.1821 |
|        Avg probs ensemble       |  0.9562  |   0.9562   | 0.7271 | 1.6108 | 0.3390 | 0.6237 | 0.0067 |  0.0067 | 0.1424 | 0.1471 |


|              Method             | Accuracy | T.Accuracy |   ECE  |  T.ECE |   MCE  |  T.MCE |  BRIER | T.BRIER |   NNL  |  T.NNL |
|:-------------------------------:|:--------:|:----------:|:------:|:------:|:------:|:------:|:------:|:-------:|:------:|:------:|
| kuangliu_densenet_best_accuracy |  0.9426  |   0.9426   | 3.3102 | 0.9455 | 1.7591 | 0.1545 | 0.0092 |  0.0087 | 0.2218 | 0.1898 |
| kuangliu_densenet_best_accuracy |  0.9401  |   0.9401   | 3.5037 | 0.9978 | 2.1417 | 0.3126 | 0.0097 |  0.0092 | 0.2404 | 0.2025 |
| kuangliu_densenet_best_accuracy |  0.9395  |   0.9395   | 3.2023 | 0.5873 | 1.8706 | 0.1562 | 0.0095 |  0.0090 | 0.2288 | 0.1947 |
| kuangliu_densenet_best_accuracy |  0.9411  |   0.9411   | 3.3464 | 0.8955 | 2.0553 | 0.2468 | 0.0095 |  0.0089 | 0.2294 | 0.1935 |
| kuangliu_densenet_best_accuracy |  0.9435  |   0.9435   | 3.0818 | 0.9934 | 1.7603 | 0.3187 | 0.0092 |  0.0087 | 0.2293 | 0.1945 |
|        Avg probs ensemble       |  0.9549  |   0.9547   | 0.7190 | 2.3787 | 0.3186 | 0.8339 | 0.0069 |  0.0070 | 0.1449 | 0.1514 |


### Benchmarks

https://benchmarks.ai/cifar-10

