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

Methods:
  - Avg ensemble: Perform an ensemble with original logits and with calibrated logits.
  - Avg ensemble T: Perform an ensemble with original logits and then learn a calibration technique.
  - Avg ensemble CT: Perform an ensemble with calibrated logits and then learn a calibration technique.

### CIFAR-10


#### Validation

|      Method     | Accuracy | T.Accuracy | M.Accuracy |   ECE  |  T.ECE |  M.ECE |   MCE  |  T.MCE |  M.MCE |  BRIER | T.BRIER | M.BRIER |   NNL  |  T.NNL |  M.NNL |
|:---------------:|:--------:|:----------:|:----------:|:------:|:------:|:------:|:------:|:------:|:------:|:------:|:-------:|:-------:|:------:|:------:|:------:|
|     resnet18    |  0.9580  |   0.9580   |   0.9578   | 1.7664 | 0.6921 | 0.7104 | 0.8891 | 0.2379 | 0.2175 | 0.0066 |  0.0064 |  0.0064 | 0.1483 | 0.1407 | 0.1397 |
|     resnet18    |  0.9560  |   0.9560   |   0.9548   | 2.1937 | 0.9936 | 0.8789 | 1.0590 | 0.2206 | 0.2016 | 0.0070 |  0.0068 |  0.0068 | 0.1594 | 0.1493 | 0.1480 |
|     resnet18    |  0.9560  |   0.9560   |   0.9556   | 2.1179 | 0.9738 | 0.9698 | 1.2142 | 0.3193 | 0.2671 | 0.0072 |  0.0070 |  0.0070 | 0.1640 | 0.1492 | 0.1482 |
|     resnet18    |  0.9554  |   0.9554   |   0.9548   | 2.0303 | 0.8682 | 0.7824 | 1.0845 | 0.2811 | 0.1482 | 0.0071 |  0.0069 |  0.0067 | 0.1603 | 0.1508 | 0.1469 |
|     resnet18    |  0.9550  |   0.9550   |    0.955   | 2.1711 | 1.1764 | 1.3017 | 1.1650 | 0.4497 | 0.3564 | 0.0073 |  0.0070 |  0.0070 | 0.1638 | 0.1531 | 0.1521 |
|   Avg ensemble  |  0.9660  |   0.9658   |    0.966   | 0.6835 | 1.4179 | 1.4143 | 0.1751 | 0.6765 | 0.6601 | 0.0053 |  0.0053 |  0.0053 | 0.1080 | 0.1126 | 0.1127 |
|  Avg ensemble T |  ------  |   0.9656   |    0.966   | ------ | 0.7155 | 0.8582 | ------ | 0.1366 | 0.2082 | ------ |  0.0053 |  0.0053 | ------ | 0.1105 | 0.1102 |
| Avg ensemble CT |  ------  |   0.9652   |   0.9662   | ------ | 0.8509 | 0.7243 | ------ | 0.1394 | 0.1679 | ------ |  0.0053 |  0.0053 | ------ | 0.1105 | 0.1099 |


|      Method     | Accuracy | T.Accuracy | M.Accuracy |   ECE  |  T.ECE |  M.ECE |   MCE  |  T.MCE |  M.MCE |  BRIER | T.BRIER | M.BRIER |   NNL  |  T.NNL |  M.NNL |
|:---------------:|:--------:|:----------:|:----------:|:------:|:------:|:------:|:------:|:------:|:------:|:------:|:-------:|:-------:|:------:|:------:|:------:|
|   wresnet28_10  |  0.9636  |   0.9636   |   0.9636   | 1.4599 | 0.7719 | 0.7240 | 0.8906 | 0.1561 | 0.1757 | 0.0059 |  0.0058 |  0.0058 | 0.1337 | 0.1272 | 0.1256 |
|   wresnet28_10  |  0.9648  |   0.9648   |   0.9642   | 1.3822 | 0.6609 | 0.6936 | 0.6119 | 0.2080 | 0.1585 | 0.0057 |  0.0056 |  0.0056 | 0.1239 | 0.1196 | 0.1189 |
|   wresnet28_10  |  0.9646  |   0.9646   |    0.964   | 1.6782 | 0.8738 | 0.5257 | 0.6925 | 0.2224 | 0.1765 | 0.0057 |  0.0055 |  0.0055 | 0.1202 | 0.1149 | 0.1133 |
|   wresnet28_10  |  0.9640  |   0.9640   |   0.9638   | 1.5114 | 0.8737 | 1.0476 | 0.5644 | 0.1712 | 0.2241 | 0.0057 |  0.0056 |  0.0056 | 0.1243 | 0.1198 | 0.1178 |
|   wresnet28_10  |  0.9654  |   0.9654   |   0.9652   | 1.1469 | 0.4461 | 0.3893 | 0.5272 | 0.2007 | 0.0985 | 0.0057 |  0.0056 |  0.0055 | 0.1268 | 0.1237 | 0.1219 |
|   Avg ensemble  |  0.9700  |   0.9694   |    0.97    | 0.7618 | 0.9937 | 1.0935 | 0.1850 | 0.4631 | 0.4842 | 0.0045 |  0.0046 |  0.0046 | 0.0913 | 0.0948 | 0.0947 |
|  Avg ensemble T |  ------  |   0.9694   |   0.9692   | ------ | 0.4403 | 0.8338 | ------ | 0.1753 | 0.1848 | ------ |  0.0046 |  0.0046 | ------ | 0.0947 | 0.0941 |
| Avg ensemble CT |  ------  |   0.9694   |   0.9696   | ------ | 0.5750 | 0.7675 | ------ | 0.1714 | 0.1764 | ------ |  0.0046 |  0.0046 | ------ | 0.0947 | 0.0940 |

#### Test

|      Method     | Accuracy | T.Accuracy | M.Accuracy |   ECE  |  T.ECE |  M.ECE |   MCE  |  T.MCE |  M.MCE |  BRIER | T.BRIER | M.BRIER |   NNL  |  T.NNL |  M.NNL |
|:---------------:|:--------:|:----------:|:----------:|:------:|:------:|:------:|:------:|:------:|:------:|:------:|:-------:|:-------:|:------:|:------:|:------:|
|     resnet18    |  0.9564  |   0.9564   |   0.9562   | 1.8005 | 0.8232 | 0.6420 | 0.8705 | 0.2252 | 0.2169 | 0.0068 |  0.0066 |  0.0066 | 0.1475 | 0.1409 | 0.1410 |
|     resnet18    |  0.9552  |   0.9552   |   0.9552   | 2.0140 | 0.7748 | 0.7228 | 0.9289 | 0.1902 | 0.2084 | 0.0070 |  0.0068 |  0.0068 | 0.1567 | 0.1483 | 0.1475 |
|     resnet18    |  0.9516  |   0.9516   |   0.9519   | 2.4544 | 1.0237 | 0.9094 | 1.2841 | 0.2751 | 0.3330 | 0.0076 |  0.0073 |  0.0073 | 0.1670 | 0.1524 | 0.1515 |
|     resnet18    |  0.9522  |   0.9522   |   0.9523   | 2.1294 | 0.8575 | 0.9576 | 1.0734 | 0.2846 | 0.3024 | 0.0075 |  0.0072 |  0.0071 | 0.1660 | 0.1562 | 0.1538 |
|     resnet18    |  0.9510  |   0.9510   |   0.9508   | 2.3753 | 0.9601 | 0.8843 | 1.0949 | 0.2478 | 0.2793 | 0.0078 |  0.0075 |  0.0075 | 0.1739 | 0.1624 | 0.1613 |
|   Avg ensemble  |  0.9635  |   0.9639   |   0.9647   | 0.5964 | 1.5580 | 1.6219 | 0.1811 | 0.6596 | 0.6415 | 0.0053 |  0.0054 |  0.0054 | 0.1096 | 0.1149 | 0.1150 |
|  Avg ensemble T |  ------  |   0.9640   |   0.9642   | ------ | 0.7664 | 0.7967 | ------ | 0.1958 | 0.2002 | ------ |  0.0053 |  0.0053 | ------ | 0.1096 | 0.1092 |
| Avg ensemble CT |  ------  |   0.9640   |   0.9644   | ------ | 0.7407 | 0.7913 | ------ | 0.2077 | 0.1914 | ------ |  0.0053 |  0.0053 | ------ | 0.1096 | 0.1092 |

|      Method     | Accuracy | T.Accuracy | M.Accuracy |   ECE  |  T.ECE |  M.ECE |   MCE  |  T.MCE |  M.MCE |  BRIER | T.BRIER | M.BRIER |   NNL  |  T.NNL |  M.NNL |
|:---------------:|:--------:|:----------:|:----------:|:------:|:------:|:------:|:------:|:------:|:------:|:------:|:-------:|:-------:|:------:|:------:|:------:|
|   wresnet28_10  |  0.9627  |   0.9627   |   0.9635   | 1.3307 | 0.5234 | 0.4685 | 0.6253 | 0.1171 | 0.0905 | 0.0058 |  0.0057 |  0.0056 | 0.1251 | 0.1215 | 0.1202 |
|   wresnet28_10  |  0.9640  |   0.9640   |   0.9641   | 1.1168 | 0.5864 | 0.6496 | 0.3270 | 0.1049 | 0.1692 | 0.0055 |  0.0054 |  0.0053 | 0.1156 | 0.1133 | 0.1125 |
|   wresnet28_10  |  0.9628  |   0.9628   |   0.9635   | 1.2869 | 0.4482 | 0.4121 | 0.5294 | 0.1254 | 0.0833 | 0.0056 |  0.0055 |  0.0055 | 0.1177 | 0.1143 | 0.1140 |
|   wresnet28_10  |  0.9656  |   0.9656   |   0.9643   | 0.9684 | 0.4047 | 0.5815 | 0.3356 | 0.1456 | 0.2221 | 0.0053 |  0.0052 |  0.0052 | 0.1129 | 0.1113 | 0.1098 |
|   wresnet28_10  |  0.9636  |   0.9636   |    0.963   | 0.9383 | 0.4526 | 0.4503 | 0.5028 | 0.0687 | 0.0836 | 0.0056 |  0.0055 |  0.0055 | 0.1228 | 0.1210 | 0.1205 |
|   Avg ensemble  |  0.9710  |   0.9710   |   0.9708   | 0.6648 | 1.4433 | 1.4274 | 0.2068 | 0.4861 | 0.5119 | 0.0043 |  0.0044 |  0.0044 | 0.0885 | 0.0928 | 0.0931 |
|  Avg ensemble T |  ------  |   0.9715   |   0.9709   | ------ | 0.4403 | 0.3278 | ------ | 0.1162 | 0.0849 | ------ |  0.0043 |  0.0043 | ------ | 0.0888 | 0.0885 |
| Avg ensemble CT |  ------  |   0.9717   |   0.9706   | ------ | 0.4675 | 0.3774 | ------ | 0.1165 | 0.0736 | ------ |  0.0043 |  0.0043 | ------ | 0.0887 | 0.0885 |


### CIFAR-100

#### Validation


|      Method     | Accuracy | T.Accuracy | M.Accuracy |   ECE  |  T.ECE |  M.ECE |   MCE  |  T.MCE |  M.MCE |  BRIER | T.BRIER | M.BRIER |   NNL  |  T.NNL |  M.NNL |
|:---------------:|:--------:|:----------:|:----------:|:------:|:------:|:------:|:------:|:------:|:------:|:------:|:-------:|:-------:|:------:|:------:|:------:|
|     resnet18    |  0.7574  |   0.7574   |   0.7604   | 5.0095 | 4.5328 | 3.8598 | 1.8547 | 1.6758 | 1.1801 | 0.0035 |  0.0035 |  0.0034 | 0.9648 | 0.9637 | 0.9431 |
|     resnet18    |  0.7762  |   0.7762   |   0.7762   | 4.4792 | 4.4686 | 3.2325 | 1.5900 | 1.6199 | 1.0169 | 0.0032 |  0.0032 |  0.0032 | 0.9153 | 0.9153 | 0.8903 |
|     resnet18    |  0.7686  |   0.7686   |    0.774   | 4.6993 | 4.8845 | 4.1583 | 1.5129 | 1.6702 | 1.1778 | 0.0033 |  0.0033 |  0.0032 | 0.9366 | 0.9362 | 0.9058 |
|     resnet18    |  0.7660  |   0.7660   |   0.7638   | 4.9267 | 4.7496 | 3.4612 | 2.0477 | 1.7208 | 1.3321 | 0.0034 |  0.0034 |  0.0033 | 0.9468 | 0.9447 | 0.9198 |
|     resnet18    |  0.7660  |   0.7660   |    0.766   | 5.9960 | 4.4861 | 3.9030 | 2.2192 | 1.5240 | 1.0452 | 0.0034 |  0.0034 |  0.0033 | 0.9404 | 0.9307 | 0.9092 |
|   Avg ensemble  |  0.8030  |   0.8040   |   0.8042   | 5.0029 | 5.6589 | 5.6573 | 0.6211 | 0.6529 | 0.6758 | 0.0028 |  0.0029 |  0.0028 | 0.7562 | 0.7617 | 0.7507 |
|  Avg ensemble T |  ------  |   0.8024   |   0.8056   | ------ | 3.8058 | 3.8232 | ------ | 1.7210 | 1.2634 | ------ |  0.0028 |  0.0027 | ------ | 0.7507 | 0.7257 |
| Avg ensemble CT |  ------  |   0.8022   |   0.8044   | ------ | 3.7844 | 3.5109 | ------ | 1.3407 | 1.2343 | ------ |  0.0028 |  0.0027 | ------ | 0.7504 | 0.7238 |

|      Method     | Accuracy | T.Accuracy | M.Accuracy |   ECE  |  T.ECE |  M.ECE |   MCE  |  T.MCE |  M.MCE |  BRIER | T.BRIER | M.BRIER |   NNL  |  T.NNL |  M.NNL |
|:---------------:|:--------:|:----------:|:----------:|:------:|:------:|:------:|:------:|:------:|:------:|:------:|:-------:|:-------:|:------:|:------:|:------:|
|   wresnet28_10  |  0.8076  |   0.8076   |   0.8104   | 5.2375 | 4.6481 | 3.3184 | 2.5213 | 1.5562 | 1.1070 | 0.0029 |  0.0028 |  0.0028 | 0.7696 | 0.7591 | 0.7359 |
|   wresnet28_10  |  0.8058  |   0.8058   |   0.8062   | 4.9362 | 4.0045 | 3.2666 | 1.8979 | 1.2335 | 0.8601 | 0.0029 |  0.0029 |  0.0028 | 0.7641 | 0.7576 | 0.7331 |
|   wresnet28_10  |   0.803  |   0.8030   |   0.8046   | 4.7467 | 4.4436 | 3.4093 | 2.1158 | 1.6936 | 1.1039 | 0.0029 |  0.0029 |  0.0028 | 0.7866 | 0.7843 | 0.7587 |
|   wresnet28_10  |  0.8056  |   0.8056   |   0.8068   | 5.9382 | 5.6757 | 3.7528 | 1.5039 | 2.0878 | 1.3159 | 0.0030 |  0.0029 |  0.0028 | 0.8239 | 0.8179 | 0.7780 |
|   wresnet28_10  |   0.805  |   0.8050   |   0.8054   | 4.7522 | 4.8595 | 3.3317 | 1.6118 | 1.8489 | 1.4331 | 0.0029 |  0.0029 |  0.0028 | 0.7921 | 0.7912 | 0.7541 |
|   Avg ensemble  |   0.832  |   0.8324   |   0.8356   | 4.0426 | 4.4469 | 4.6226 | 0.5341 | 0.5511 | 0.6020 | 0.0025 |  0.0025 |  0.0024 | 0.6326 | 0.6347 | 0.6190 |
|  Avg ensemble T |  ------  |   0.8312   |    0.836   | ------ | 3.6406 | 2.9640 | ------ | 1.4949 | 1.1989 | ------ |  0.0025 |  0.0024 | ------ | 0.6339 | 0.6046 |
| Avg ensemble CT |  ------  |   0.8320   |   0.8362   | ------ | 3.6419 | 2.9996 | ------ | 1.5072 | 1.2052 | ------ |  0.0025 |  0.0023 | ------ | 0.6354 | 0.6040 |

#### Test

|      Method     | Accuracy | T.Accuracy | M.Accuracy |   ECE  |  T.ECE |  M.ECE |   MCE  |  T.MCE |  M.MCE |  BRIER | T.BRIER | M.BRIER |   NNL  |  T.NNL |  M.NNL |
|:---------------:|:--------:|:----------:|:----------:|:------:|:------:|:------:|:------:|:------:|:------:|:------:|:-------:|:-------:|:------:|:------:|:------:|
|     resnet18    |  0.7571  |   0.7571   |   0.7537   | 4.6541 | 4.5502 | 4.2280 | 1.6328 | 1.4557 | 1.4476 | 0.0034 |  0.0034 |  0.0034 | 0.9571 | 0.9571 | 0.9682 |
|     resnet18    |  0.7697  |   0.7697   |   0.7672   | 4.7226 | 4.7144 | 4.2650 | 1.6386 | 1.6928 | 1.2600 | 0.0033 |  0.0033 |  0.0033 | 0.9227 | 0.9226 | 0.9274 |
|     resnet18    |  0.7622  |   0.7622   |   0.7603   | 4.5735 | 4.7826 | 4.4034 | 1.5450 | 1.6766 | 1.5103 | 0.0034 |  0.0034 |  0.0034 | 0.9560 | 0.9561 | 0.9602 |
|     resnet18    |  0.7654  |   0.7654   |   0.7637   | 4.9192 | 4.1652 | 3.7157 | 1.7126 | 1.3871 | 1.2789 | 0.0033 |  0.0033 |  0.0033 | 0.9271 | 0.9258 | 0.9272 |
|     resnet18    |  0.7602  |   0.7602   |   0.7591   | 5.6894 | 4.0721 | 3.8514 | 1.8701 | 1.1319 | 1.0822 | 0.0034 |  0.0034 |  0.0034 | 0.9240 | 0.9193 | 0.9239 |
|   Avg ensemble  |  0.8009  |   0.8019   |   0.8019   | 5.2226 | 5.7798 | 5.9095 | 0.6142 | 0.6458 | 0.7102 | 0.0028 |  0.0029 |  0.0029 | 0.7551 | 0.7611 | 0.7688 |
|  Avg ensemble T |  ------  |   0.8038   |   0.8034   | ------ | 3.7720 | 4.0493 | ------ | 1.3567 | 1.4328 | ------ |  0.0028 |  0.0028 | ------ | 0.7416 | 0.7474 |
| Avg ensemble CT |  ------  |   0.8043   |   0.8037   | ------ | 3.7684 | 4.0441 | ------ | 1.3407 | 1.4558 | ------ |  0.0028 |  0.0028 | ------ | 0.7419 | 0.7476 |

|      Method     | Accuracy | T.Accuracy | M.Accuracy |   ECE  |  T.ECE |  M.ECE |   MCE  |  T.MCE |  M.MCE |  BRIER | T.BRIER | M.BRIER |   NNL  |  T.NNL |  M.NNL |
|:---------------:|:--------:|:----------:|:----------:|:------:|:------:|:------:|:------:|:------:|:------:|:------:|:-------:|:-------:|:------:|:------:|:------:|
|   wresnet28_10  |  0.8076  |   0.8076   |   0.8039   | 4.2877 | 2.6976 | 2.7425 | 1.6742 | 1.0555 | 1.0330 | 0.0028 |  0.0028 |  0.0028 | 0.7446 | 0.7391 | 0.7461 |
|   wresnet28_10  |  0.7994  |   0.7994   |   0.8021   | 4.6489 | 3.3730 | 2.8870 | 1.7562 | 1.1220 | 1.0527 | 0.0029 |  0.0028 |  0.0028 | 0.7623 | 0.7579 | 0.7552 |
|   wresnet28_10  |  0.8059  |   0.8059   |   0.8017   | 4.5047 | 4.2147 | 3.9155 | 1.6693 | 1.4077 | 1.2652 | 0.0028 |  0.0028 |  0.0028 | 0.7642 | 0.7632 | 0.7668 |
|   wresnet28_10  |  0.8075  |   0.8075   |   0.8015   | 4.8194 | 4.5525 | 3.7177 | 1.0045 | 1.5004 | 1.4120 | 0.0028 |  0.0028 |  0.0028 | 0.7969 | 0.7849 | 0.7896 |
|   wresnet28_10  |  0.8066  |   0.8066   |   0.8066   | 3.5695 | 3.6668 | 3.5561 | 1.0873 | 1.3088 | 1.2132 | 0.0028 |  0.0028 |  0.0028 | 0.7662 | 0.7642 | 0.7656 |
|   Avg ensemble  |   0.835  |   0.8352   |   0.8341   | 4.2427 | 4.6349 | 4.8263 | 0.5986 | 0.6204 | 0.5634 | 0.0024 |  0.0024 |  0.0024 | 0.6241 | 0.6263 | 0.6323 |
|  Avg ensemble T |  ------  |   0.8359   |   0.8311   | ------ | 2.7199 | 2.7166 | ------ | 1.0560 | 1.1645 | ------ |  0.0024 |  0.0024 | ------ | 0.6135 | 0.6197 |
| Avg ensemble CT |  ------  |   0.8361   |   0.8314   | ------ | 2.9168 | 2.8049 | ------ | 1.0968 | 1.1532 | ------ |  0.0024 |  0.0024 | ------ | 0.6142 | 0.6203 |

### Benchmarks

https://benchmarks.ai/cifar-10
https://benchmarks.ai/cifar-100

### Credits

  - RandAugment Pytorch Implementation: https://github.com/ildoonet/pytorch-randaugment
