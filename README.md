# Time aware and Feature similarity Self Attention in Vessel Fuel Consumption Prediction

This is a Pytorch implementation of [Time aware and Feature similarity Self Attention in Vessel Fuel Consumption Prediction](https://www.mdpi.com/2076-3417/11/23/11514). This repository only addresses the models when backbone is BiLSTM. It is possible to change the backbones with class in models.py. Each attention moudels is located before the prediction layer. Ensemble consists of three fully-connected layers. 
\
The overall architecture and attention process are as below:

<center><img src="./images/Model Process.jpg" width="70%" height="70%"></center>

<center><img src="./images/Attention Process.jpg" width="30%" height="30%"></center>

\
The models are designed to consider data properties, (i.e., sequential, irregular time, feature importance). The repository supports BiLSTM (BASE), Self-Attention (SA), Time-aware Attention (TA), Feature-similarity Attention (FA), Ensemble (ENS). SA is the implementation of self-attention from Transformer. Each TA and FA takes a process to combine with SA befor making attention weights. Note that the repositoy does not support dataset.
 
## Requirments

python == 3.7.4 \
pytorch == 1.7.1

## Usage

### 1. train model
<pre>
<code>
python main.py train --att_type [attention type]
</code>
</pre>

### 2. test model
<pre>
<code>
python main.py test --att_type [attention type]
</code>
</pre>

### 3. parameter settings
#### Possible to change parameters including paths, when you call main.py.

* epoch: 100
* batch_size: 200
* lr: 0.001

* att_type: BASE / SA / TA / FA / ENS
* hidden_size: 4
* sequence_length: 50
* num_layers: 2
* dropout: 0.1
* function_type: sig


## Visualizations

### 1. Feature importance
#### Estimated feature importance from FA model. Feature importance is learnable parameters.

<center><img src="./images/Importance.jpg" width="40%" height="40%"></center>

### 2. TA attention map
#### One example of TA attention map. TA ignores point where time difference increases rapidly.

<center><img src="./images/TA_MAP.jpg" width="40%" height="40%"></center>


### 3. FA attention map
#### One example of FA attention map. FA makes symmetric attention maps and FA is especially affected by Speed.

<center><img src="./images/FA_MAP.jpg" width="40%" height="40%"></center>
