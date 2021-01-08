# Ensemble Analysis

### Requirements

Paciencia++

Pytoch >= 1.7


# Procedure

For ensembling first generate the logits with evaluate. Next, move all the desired logits to a directory 
and pass it through argument to `ensemble.py`.

### ToDo

- SWAG

## Results

### CIFAR-10

#### Validation

|             Method           | Accuracy | T.Accuracy |   ECE  |  T.ECE |   MCE  |  T.MCE |  BRIER | T.BRIER |   NNL  |  T.NNL |
|:----------------------------:|:--------:|:----------:|:------:|:------:|:------:|:------:|:------:|:-------:|:------:|:------:|
| cifar_resnet18_best_accuracy |  0.9478  |   0.9478   | 2.4941 | 0.9427 | 1.2195 | 0.1853 | 0.0081 |  0.0078 | 0.1915 | 0.1756 |
| cifar_resnet18_best_accuracy |  0.9488  |   0.9488   | 2.6258 | 0.9098 | 1.5179 | 0.2069 | 0.0083 |  0.0080 | 0.1925 | 0.1724 |
| cifar_resnet18_best_accuracy |  0.9500  |   0.9500   | 2.6483 | 1.0319 | 1.3220 | 0.2428 | 0.0081 |  0.0077 | 0.1883 | 0.1720 |
| cifar_resnet18_best_accuracy |  0.9472  |   0.9472   | 2.9554 | 1.0328 | 1.5557 | 0.2270 | 0.0086 |  0.0081 | 0.2001 | 0.1787 |
| cifar_resnet18_best_accuracy |  0.9490  |   0.9490   | 2.6576 | 0.9164 | 1.4270 | 0.2222 | 0.0083 |  0.0079 | 0.1937 | 0.1770 |
|       Avg probs ensemble     |  0.9582  |   0.9586   | 0.8310 | 1.5347 | 0.3613 | 0.6025 | 0.0065 |  0.0065 | 0.1394 | 0.1425 |


#### Test

|             Method           | Accuracy | T.Accuracy |   ECE  |  T.ECE |   MCE  |  T.MCE |  BRIER | T.BRIER |   NNL  |  T.NNL |
|:----------------------------:|:--------:|:----------:|:------:|:------:|:------:|:------:|:------:|:-------:|:------:|:------:|
| cifar_resnet18_best_accuracy |  0.9482  |   0.9482   | 2.4885 | 0.8139 | 1.3318 | 0.1761 | 0.0082 |  0.0079 | 0.1930 | 0.1774 |
| cifar_resnet18_best_accuracy |  0.9438  |   0.9438   | 2.9105 | 0.7971 | 1.5244 | 0.2085 | 0.0090 |  0.0086 | 0.2085 | 0.1858 |
| cifar_resnet18_best_accuracy |  0.9469  |   0.9469   | 2.5997 | 0.8008 | 1.3823 | 0.1965 | 0.0084 |  0.0080 | 0.1900 | 0.1754 |
| cifar_resnet18_best_accuracy |  0.9448  |   0.9448   | 2.7649 | 0.7183 | 1.4504 | 0.3047 | 0.0088 |  0.0084 | 0.2045 | 0.1843 |
| cifar_resnet18_best_accuracy |  0.9458  |   0.9458   | 2.6418 | 0.7023 | 1.4205 | 0.1694 | 0.0085 |  0.0082 | 0.1966 | 0.1821 |
|       Avg probs ensemble     |  0.9562  |   0.9562   | 0.7271 | 1.6108 | 0.3390 | 0.6237 | 0.0067 |  0.0067 | 0.1424 | 0.1471 |


### CIFAR-100

#### Validation

|            Method            | Accuracy | T.Accuracy |   ECE  |  T.ECE |   MCE  |  T.MCE |  BRIER | T.BRIER |   NNL  |  T.NNL |
|:----------------------------:|:--------:|:----------:|:------:|:------:|:------:|:------:|:------:|:-------:|:------:|:------:|
| cifar_resnet18_best_accuracy |  0.7594  |   0.7594   | 6.5547 | 5.9802 | 2.7422 | 2.0106 | 0.0035 |  0.0035 | 0.9940 | 0.9879 |
| cifar_resnet18_best_accuracy |   0.753  |    0.753   | 6.0191 | 5.8687 | 2.4787 | 2.1335 | 0.0036 |  0.0036 | 1.0402 | 1.0383 |
| cifar_resnet18_best_accuracy |  0.7568  |   0.7568   | 6.0899 | 5.6402 | 2.1505 | 2.0791 | 0.0035 |  0.0035 | 0.9972 | 0.9970 |
| cifar_resnet18_best_accuracy |  0.7648  |   0.7648   | 6.1220 | 5.9415 | 1.9955 | 1.9311 | 0.0034 |  0.0034 | 0.9784 | 0.9780 |
| cifar_resnet18_best_accuracy |  0.7548  |   0.7548   | 6.5176 | 5.7134 | 2.3225 | 1.9573 | 0.0035 |  0.0035 | 1.0059 | 1.0033 |
|      Avg probs ensemble      |   0.79   |   0.7908   | 4.6465 | 5.2404 | 0.5740 | 0.6919 | 0.0030 |  0.0030 | 0.8194 | 0.8270 |

#### Test

|            Method            | Accuracy | T.Accuracy |   ECE  |  T.ECE |   MCE  |  T.MCE |  BRIER | T.BRIER |   NNL  |  T.NNL |
|:----------------------------:|:--------:|:----------:|:------:|:------:|:------:|:------:|:------:|:-------:|:------:|:------:|
| cifar_resnet18_best_accuracy |  0.7575  |   0.7575   | 5.7407 | 4.9287 | 2.0491 | 1.5366 | 0.0034 |  0.0034 | 0.9818 | 0.9786 |
| cifar_resnet18_best_accuracy |  0.7438  |   0.7438   | 5.9989 | 5.2826 | 2.2892 | 1.9045 | 0.0036 |  0.0036 | 1.0409 | 1.0390 |
| cifar_resnet18_best_accuracy |  0.7577  |   0.7577   | 5.0977 | 5.0630 | 1.9965 | 1.9169 | 0.0034 |  0.0034 | 0.9841 | 0.9840 |
| cifar_resnet18_best_accuracy |  0.7591  |   0.7591   | 5.1461 | 4.9907 | 2.0688 | 2.0140 | 0.0034 |  0.0034 | 0.9829 | 0.9826 |
| cifar_resnet18_best_accuracy |  0.7593  |   0.7593   | 5.3094 | 4.7879 | 2.2315 | 1.9431 | 0.0034 |  0.0034 | 0.9912 | 0.9899 |
|      Avg probs ensemble      |  0.7909  |   0.7903   | 4.4063 | 5.0176 | 0.4668 | 0.5333 | 0.0030 |  0.0030 | 0.8153 | 0.8234 |


### Benchmarks

https://benchmarks.ai/cifar-10
https://benchmarks.ai/cifar-100

### Credits

  - RandAugment Pytorch Implementation: https://github.com/ildoonet/pytorch-randaugment