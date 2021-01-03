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
| Res18 | Seed 311220201 |   0.9431   | 0.9473 |
| Res18 | Seed 311220202 |   0.9441   | 0.9437 |
| Res18 | Seed 311220203 |   0.9482   | 0.9467 |
| Res18 | Seed 311220204 |   0.9425   | 0.9447 |
| Res18 | Seed 311220205 |   0.9414   | 0.9457 |


### Benchmarks

https://benchmarks.ai/cifar-10

