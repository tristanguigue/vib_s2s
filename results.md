# Results

## FeedForward
### 1
Batch = 2048
Samples = 12
Setup: check VIB
- Beta = 0: 98.26, 0.064
- Beta = 10^-4: 98.52, 0.055
- Beta = 5.10^-4, 1501936288: 98.62, 0.053

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
- Beta = 10^-3, 1502030824: 0.0253, 99.10
- Beta = 10^-2, 1502045656 (not finished)


## Seq2Seq MNIST Binary
### 1 Small dataset *Works*
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

### 3 All data
Length = 60
Start = 300
Train = All
Test = All
Hidden = 128
Bottleneck = 32
Batch = 1000
Samples = 12
- Beta = 0, 1502100753: 0.1439, 94.8559, no overfit
- Beta = 10^-3, 1502102918: 0.1443, 94.852
- Beta = 10^-1, 1502104514: 0.1477, 94.80
- Beta = 10^-2, 1502105550: 0.1451, 94.82

### 4 Longer sequence
Length = 120
Start = 300
Train = All
Test = All
Hidden = 128
Bottleneck = 32
Batch = 1000
Samples = 12
- Beta = 0, 1502100753: 0.237, 91.637, no overfit

### 5 Larger bottleneck
Length = 60
Start = 300
Train = All
Test = All
Hidden = 128
Bottleneck = 64
Batch = 2048
Samples = 12
- Beta = 0, 1502100753: 0.1432, 94.898, no Overfit
- Beta=10^-5, 1502117681: 0.1431, 94.92
- Beta=10^-3, 1502129328, Long run: 0.1432, 94.903, Very mild Overfit

### 6 Sample loss
Length = 60
Start = 300
Train = All
Test = All
Hidden = 128
Bottleneck = 32
Batch = 2048
Samples = 12
- Beta=0, 1502119986: 0.2244, 88.159

### 7 5k data with large bottleneck
Length = 60
Start = 300
Train = 5k
Test = All
Hidden = 128
Bottleneck = 64
Batch = 512
Samples = 12
- Beta=0: 94.84, 0.1450, no overfit

### 8 1k data
Length = 60
Start = 300
Train = 1k
Test = All
Hidden = 128
Bottleneck = 64
Batch = 500
Samples = 12
- Beta=0: 94.75, 0.1525, no overfit

### 9 GRU
Length = 60
Start = 300
Train = All
Test = All
Hidden = 128
Bottleneck = 32
Batch = 1000
Samples = 12
- Beta=0: 97.53, 0.059, no clear overfit (2323 epochs)
- Beta=10^-3:  97.498, 0.063

## Seq2seq Continous
### 1 Cubic + sinusoidal + noise
Length = 60
Train = 5k
Test = 2k
Hidden = 128
Bottleneck = 32
Batch = 500
Samples = 12
- Beta = 0, 1502184709: 0.0493, no overfit
- Beta = 10^-3, 1502196521

## Seq2Seq Binary Generated
### 1
Length = 60
Train = 5k
Test = 2k
Hidden = 128
Bottleneck = 32
Batch = 500
Samples = 12
- Beta = 0: 82.36, 0.444, overfit yes
- Beta = 10^-3: 82.03, 0.451

## Seq2Labels
### 1
Length = 5
Train = All
Test = All
Hidden = 128
Bottleneck = 32
Batch = 500
Samples = 12
Rate = 10^-4
- Beta= 10^-4, 1502264047: 33.79, 1.880
- Beta = 10^-3, 1502222038: 33.569, 1.8775
- Beta = 10^-2, 1502237492: 32.96, 1.893

### 2 Smaller learning rate
Length = 5
Train = All
Test = All
Hidden = 128
Bottleneck = 32
Batch = 500
Samples = 12
Rate = 2*10^-5
- Beta = 0, 1502265059: 33.80, 1.889
- Beta= 10^-4, 1502269300: 33.81, 1.885

### 3 Larger Batch *Works*
Length = 5
Train = All
Test = All
Hidden = 128
Bottleneck = 32
Batch = 2000
Samples = 12
Rate = 2*10^-5
- Beta=0: 33.33, 1.893
- Beta = 10^-3: 33.77, 1.8741
- Beta = 10^-4: 33.76, 1.8779
- Beta = 10^-2: 33.58, 1.895

### 4 Varying bottleneck and decoder hidden units *Works*
Length = 5
Train = All
Test = All
Hidden = 128
Batch = 2000
Samples = 12
Rate = 2*10^-5
- Beta = 0, bottleneck = 16, hidden2 = 16: 1.896, 32.88
- Beta = 10^-3, bottleneck = 16, hidden2 = 16: 1.871, 33.94
- Beta = 10^-3, bottleneck = 32, hidden2 = 16: 1.88, 33.43
- Beta = 10^-3, bottleneck = 8, hidden2 = 8: 1.884, 33.07
- Beta = 10^-2, bottleneck = 8, hidden2 = 8: 1.887, 32.89
- Beta = 10^-3, bottleneck = 16, hidden2 = 8: 1.884, 31.709

### 5 Longer Sequence *Works*
Length = 10
Train = All
Test = All
Hidden = 128
Bottleneck = 32
Hidden decoder = 32
Batch = 2000
Samples = 12
Rate = 2*10^-5
- Beta = 0: 2.128, 22.29
- Beta = 10^-3: 2.115, 22.49

### 6 Updating marginal
Length = 5
Train = All
Test = All
Hidden = 128
Bottleneck = 32
Hidden decoder = 32
Batch = 2000
Samples = 12
Rate = 2*10^-5
Update marginal = True
- Beta = 10^-3: 1.87, 33.11

### 7 Varying Hidden state
Length = 5
Train = All
Test = All
Bottleneck = 16
Hidden decoder = 16
Batch = 2000
Samples = 12
Rate = 2*10^-5
Update marginal = False
- Beta = 10^-3, Hidden = 256: 1.869, 32.88
- Beta = 0, Hidden = 64: 1.901, 32.71

### 8 GRU
Length = 5
Train = All
Test = All
Hidden = 128
Batch = 2000
Samples = 12
Rate = 3*10^-5
Bottleneck = 16
Hidden2 = 16
- Beta 0: 69.55, 0.936| 66.93, 1.008
- Beta 10^-4: 68.07, 0.99
- Beta 5.10^-4: 68.77, 1.03
- Beta 10^-3: 73.55, 0.9555 | 69.549, 1.037
- Beta: 2.10^-3: 
- Beta: 5.10^-3: 70.72, 1.136
- Beta 10^-2: 68.3, 1.26

### 9 GRU Larger bottleneck
Length = 5
Train = All
Test = All
Batch = 2000
Samples = 12
Rate = 3*10^-5
- Hidden = 128, bottleneck = 32, hidden2 = 32
    - Beta=0: 79.93, 0.669
    - Beta=10^-3: 81.78, 0.742
- Hidden = 256, bottleneck = 128, hidden2 = 126
    - Beta=0: 87.38, 0.439
    - Beta=10^-3: 91.1, 0.502

### 10 GRU Even larger bottleneck
Length = 5
Train = All
Test = All
Batch = 2000
Samples = 12
Rate = 3*10^-5
- Hidden = 1024, bottleneck = 512, hidden2 = 512
    - Beta=0: 90.36, 0.3471
- Hidden = 512, bottleneck = 512, hidden2 = 512
    - Beta=0: 90.47, 0.3662
- Hidden = 512, bottleneck = 256, hidden2 = 256
    - Beta=0: 89.6, 0.378
    - Beta=10^-6: 91.47, 0.385
    - Beta=10^-5: 92.82, 0.39
    - Beta=5.10^-5: 92.97, 0.4025
    - Beta=10^-4: 92.65, 0.4159
    - Beta=5.10^-4: 92.00, 0.4630
    - Beta=10^-2: 84.17, 0.895

## Seq2Labels CNN
### 1
Length = 5
Train = All
Test = All
Hidden = 128
Bottleneck = 32
Batch = 200
Samples = 12
Rate = 2*10^-5
- Beta = 0, 1502280230: 1.845, 29.43, no overfit
- Beta = 10^-3: 1.848, 29.27

### 2 GRU
`nohup python3 -u s2s_imdigit_cnn.py --batch=2000 --length=5 --samples=12 --rate=0.00003 --hidden1=512 --beta=0 --hidden2=256 --bottleneck=256 --epochs=10000 > logs/s2s_imgdigitcnn_beta0_gru &`
Length = 5
Train = All
Test = All
Batch = 2000
Samples = 12
Rate = 3*10^-5
Hidden1 = 512
Bottleneck = 256
Hidden2 = 256
Beta = 0: 0.083, 97.65, no overfit after 1000 epochs
Beta = 10^-3: 0.12, 97.61

## Seq2Label
### 1 
Length = 5
Train = All
Test = All
Hidden = 128
Bottleneck = 32
Batch = 500
Samples = 12
Rate = 10^-4
- Beta = 0: 0.313, 93.95, yes overfit
- Beta = 10^-3: 0.27, 95.35

## Videos
### 1
Batch=100
Train = 400
Test = 100
Samples = 1
Bottleneck = 32
Hidden = 512
Input size = 36 * 36
Learning rate=5.10^-5
- Beta=0: 0.054, no overfit after 10k
- Beta=10^-4: 0.0438, overfit around 4k epochs
- Beta=5.10^-4:0.04849
- Beta=10^-3: 0.05104

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

