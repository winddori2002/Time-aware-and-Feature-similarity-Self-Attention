# Time aware and Feature similarity Self Attention in Vessel Fuel Consumption Prediction

This is a Pytorch implementation of the Time aware and Feature similarity Self Attention in Vessel Fuel Consumption Prediction. asd

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
