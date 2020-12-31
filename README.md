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

#### CIFAR-10

| Model |      Info      | Validation |  Test  |
|:-----:|:--------------:|:----------:|:------:|
| Res18 | Seed 301220201 |   0.9500   | 0.9481 |
| Res18 | Seed 301220202 |   0.9478   | 0.9455 |
| Res18 | Seed 301220203 |   0.9482   | 0.9465 |
| Res18 | Seed 301220204 |   0.9427   | 0.9484 |
| Res18 | Seed 301220205 |   0.9501   | 0.9464 |


### Benchmarks

https://benchmarks.ai/cifar-10

