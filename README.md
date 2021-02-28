# Time aware and Feature similarity Self Attention in Vessel Fuel Consumption Prediction

This is a Pytorch implementation of the Time aware and Feature similarity Self Attention in Vessel Fuel Consumption Prediction. This repository only addresses the models when backbone is BiLSTM. It is possible to change the backbones with class in models.py. 

<center><img src="./images/Model Process.jpg" width="70%" height="70%"></center>

<center><img src="./images/Attention Process.jpg" width="70%" height="70%"></center>

\
The models are designed to consider data properties, (i.e., sequential, irregular time, feature importance). The repository supports BiLSTM (BASE), Self-Attention (SA), Time-aware Attention (TA), Feature-similarity Attention (FA), Ensemble (ENS). Note that the repositoy does not support dataset.
 
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
