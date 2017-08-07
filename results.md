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


## Seq2Seq
### 1 5k data
Length = 60
Start = 300
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

# 2 All data
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

# 3 Longer sequence
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

# 4 Larger bottleneck
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

- Beta=0, learning rate = 10^-5, 1502117681
Loss:  0.1431
Accuracy: 94.92
Overfit: No

# 5 Sample loss
Length = 60
Start = 300
Train = All
Test = All
Hidden = 128
Bottleneck = 32
Batch = 2048
Samples = 12


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

