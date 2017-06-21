## Reinforcement learning and attention

- Soft attention: take expectation (easier, more expensive) vs hard attention: sample (using RL)
- RNNs are Turing-Complete

### Reccurent models for visual attention
June 14, Mnih, Hees, Graves: https://arxiv.org/pdf/1406.6247.pdf

Extracting information from an image or video by adaptively selecting a sequence of regions or locations and only processing the selected regions at high resolution.

Non-differentiable: trained with reinforcement learning (policy gradient).

RNN that selects the next location in image to attend to based on past information and the demands of the task.

### Neural Turing Machines
October 14, Graves, Wayne, Danihelka: https://arxiv.org/pdf/1410.5401.pdf

Enrich RNN with large, addressable memory

### End-to-End Memory Networks
Sukhbaatar et. al. (Nov 2015)

Recurrent attention model reads from a possibly large external memory multiple times before outputting a symbol. Memory network required supervision at each layer, this is differentiable.



### DRAW
Gregor et. al. (2015)


### Teaching Machines to Read and Comprehend
Hermann et. al. (2015)


### Other papers
- Generating Sequences with Recurrent Neural Networks, Graves (2013)
- Neural Machine Translation by Jointly Learning to Align and Translate, Bahdanau et. al. (2014)
- Spatial Transformer Networks (Jaderberg et. al. 2015) 
- Neural Machine Translation in Linear Time,
Kalchbrenner et. al. (2016) 
- Differentiable Neural Computers: Hybrid computing using a neural network with dynamic external memory, Graves et. al. (2016)