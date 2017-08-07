# Results

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
### 1
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
No significan difference
- Beta = 10^-2, 1501951080
No significan difference
- Beta = 1, 1501952853
Loss: 0.1490 (from graph, smoothed 0.8)
Accuracy: 0.1490

# 2
Length = 60
Start = 300
Train = All
Test = All
Hidden = 128
Bottleneck = 32
Batch = 1000
Samples = 12

-Beta = 0
Loss: 0.14422
Accuracy: 94.8473
Overfit: a bit after 500 epochs

## Sigma
### 1
On feed forward
- Beta = 0
Start at 0.68, down to 0.50 then back up to 1
- Beta = 1
Down to 0.02
