## Reinforcement learning and attention

- Soft attention: take expectation (easier, more expensive) vs hard attention: sample (using RL)
- RNNs are Turing-Complete

### Reccurent models for visual attention
Mnih, Hees, Graves (June 2014) https://arxiv.org/pdf/1406.6247.pdf

Extracting information from an image or video by adaptively selecting a sequence of regions or locations and only processing the selected regions at high resolution.

Non-differentiable: trained with reinforcement learning (policy gradient).

RNN that selects the next location in image to attend to based on past information and the demands of the task.

### Neural Turing Machines
Graves, Wayne, Danihelka (Dec 2014) https://arxiv.org/pdf/1410.5401.pdf

Enrich RNN with large, addressable memory

### End-to-End Memory Networks
Sukhbaatar et. al. (Nov 2015) https://arxiv.org/pdf/1503.08895.pdf

Recurrent attention model reads from a possibly large external memory multiple times before outputting a symbol. Memory network required supervision at each layer, this is differentiable.

Each sentence is embedded twice, first embedding is combined with question embedding to generate weights. Weights are then combined with second output to create output. Several hops / layers of those update an internal state.


### DRAW
Gregor et. al. (2015)

Instead of creating an image instantly, it uses a recurrent neural network as both the encoder and decoder portions of a typical variational autoencoder. Every timestep, a new latent code is passed from the encoder to the decoder.

### Teaching Machines to Read and Comprehend
Hermann et. al. (Nov 2015)

New database of document–query–answer triples (answer is word ommited in summary of article) to do supervised instead of unsupervised approaches.

Used generalisation of the application of Memory Networks to question answering.

# Hybrid computing using a neural network with dynamic external memory
Graves, Wayne et. al. (2016)

Introduces differentiable neural computer (DNC). Combine the advantages of neural and computational processing by providing a neural network with read–write access to external memory. Access to memory is narrowly focused, minimizing interference among memoranda and enabling long-term storage. DNC uses differentiable attention mechanisms to access memory.

We keep track of consecutively written locations in atemporal link matrix which gives ability to recover sequences.

Memory can be searched based on the content of each location, or the associative temporal links can be followed forward and backward to recall information written in sequence or in reverse.

### Other papers
- Generating Sequences with Recurrent Neural Networks, Graves (2013)
- Neural Machine Translation by Jointly Learning to Align and Translate, Bahdanau et. al. (2014)
- Spatial Transformer Networks (Jaderberg et. al. 2015) 
- Neural Machine Translation in Linear Time,
Kalchbrenner et. al. (2016) 
- Differentiable Neural Computers: Hybrid computing using a neural network with dynamic external memory, Graves et. al. (2016)