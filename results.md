# Results

## FeedForward
### 1
Batch = 2048
Samples = 12
Setup: check VIB

- Beta = 0
Accuracy = 98.26
Loss = 0.064
- Beta = 10^-4
Accuracy = 98.52
Loss = 0.055
- Beta = 5.10^-4, 1501936288
Loss = 0.053
Accuracy = 98.62 / error 1.38


## Seq2Pixel
### 1
Length = 60
Start = 300
Train = All
Test = All
Hidden = 64
Bottleneck = 16
Batch = 1k
Samples = 1

- Beta = 0, 1502012284
Loss: 0.0235
Accuracy: 99.13
Overfit: yes, after 1500 epochs
- Beta = 10^-3, 1502030824
Loss: 0.0253
Accuracy: 99.10
- Beta = 10^-2, 1502045656 (not finished)


## Seq2Seq Binary
### 1 Small dataset, IB works
Start = 0
Length = 261
Hidden = 64
Bottleneck = 32
Batch = 100
Train = 500
Test = 500
Predict = 15

- Beta = 0, 1501753687
Loss: 0.3903 (smoothed 0.9)
Overfit: yes after 4k epochs

- Beta = 10^-2, 1501685818
Loss: 0.3716 (smoothed 0.9)

### 2 5k data
Length = 60
Start = 300
Predict = 15
Train = 5k
Test = 1k
Hidden = 128
Bottleneck = 32
Batch = 500
Samples = 12

- Beta=0, 1501945040
Loss: 0.14135 (from graph, smoothed 0.8)
Accuracy: 95.12
Overfit: A bit after 3k epochs
- Beta = 10^-4, 1501949516
No significant difference
- Beta = 10^-2, 1501951080
No significant difference
- Beta = 1, 1501952853
Loss: 0.1490 (from graph, smoothed 0.8)
Accuracy: 0.1490

# 3 All data
Length = 60
Start = 300
Train = All
Test = All
Hidden = 128
Bottleneck = 32
Batch = 1000
Samples = 12

- Beta = 0, 1502100753
Loss: 0.1439
Accuracy: 94.8559
Overfit: no
- Beta = 10^-3, 1502102918
Loss:  0.1443
Accuracy: 94.852
- Beta = 10^-1, 1502104514
Loss: 0.1477
Accuracy: 94.80
- Beta = 10^-2, 1502105550
Loss: 0.1451
Accuracy: 94.82

# 4 Longer sequence
Length = 120
Start = 300
Train = All
Test = All
Hidden = 128
Bottleneck = 32
Batch = 1000
Samples = 12

- Beta = 0, 1502100753
Loss: 0.237
Accuracy: 91.637
Overfit: no

# 5 Larger bottleneck
Length = 60
Start = 300
Train = All
Test = All
Hidden = 128
Bottleneck = 64
Batch = 2048
Samples = 12

- Beta = 0, 1502100753
Loss: 0.1432
Accuracy: 94.898
Overfit: no

- Beta=10^-5, 1502117681
Loss:  0.1431
Accuracy: 94.92
Overfit: No

- Beta=10^-3, 1502129328, Long run
Loss: 0.1432
Accuracy: 94.903
Overfit: Very mild

# 6 Sample loss
Length = 60
Start = 300
Train = All
Test = All
Hidden = 128
Bottleneck = 32
Batch = 2048
Samples = 12

- Beta=0, 1502119986
Loss: 0.2244
Accuracy: 88.159

# 7 5k data with large bottleneck
Length = 60
Start = 300
Train = 5k
Test = All
Hidden = 128
Bottleneck = 64
Batch = 512
Samples = 12

- Beta=0
Loss: 0.1450
Accuracy: 94.84
Overfit: No

# 8 1k data
Length = 60
Start = 300
Train = 1k
Test = All
Hidden = 128
Bottleneck = 64
Batch = 500
Samples = 12

- Beta=0
Loss: 0.1525
Accuracy: 94.75
Overfit: No


## Seq2seq Continous
# 1 Cubic + sinusoidal + noise
Length = 60
Train = 5k
Test = 2k
Hidden = 128
Bottleneck = 32
Batch = 500
Samples = 12

- Beta = 0, 1502184709
Loss = 0.0493
Overfit: No

- Beta = 10^-3, 1502196521
Loss = 

## Seq2Labels
# 1
Length = 5
Train = All
Test = All
Hidden = 128
Bottleneck = 32
Batch = 500
Samples = 12
Rate = 10^-4

- Beta = 0, rate=2*10^-5, 1502265059
Accuracy: 33.80
Loss: 1.889

- Beta= 10^-4, 1502264047
Accuracy: 33.79
Loss: 1.880

- Beta= 10^-4, rate=2*10^-5, 1502269300
Accuracy: 33.81
Loss: 1.885

- Beta = 10^-3, 1502222038
Accuracy: 33.569
Loss: 1.8775

- Beta = 10^-3, rate=2*20^-5, batch=2000
Loss: 1.874
Accuracy: 33.77

- Beta = 10^-2, 1502237492
Accuracy: 32.96
Loss: 1.893

## Sigma
### 1
On feed forward
- Beta = 0
Start at 0.68, down to 0.50 then back up to 1
- Beta = 1
Down to 0.02

## Compare low K with high beta
- K = 256, beta = 0
Accuracy = 98.26
Loss = 0.064
- K = 256, beta = 5.10^-4
Loss = 0.053
Accuracy = 98.62 / error 1.38
- K = 32, beta = 0
Loss: 0.066
Accuracy: 98.09
- K = 64, beta = 0
Loss: 0.067
Accuracy: 98.12
- K = 128, beta=0
Loss:0.0655
Accuracy: 98.14

