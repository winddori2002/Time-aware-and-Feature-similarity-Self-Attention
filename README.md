# Time aware and Feature similarity Self Attention in Vessel Fuel Consumption Prediction

This is a Pytorch implementation of [Time aware and Feature similarity Self Attention in Vessel Fuel Consumption Prediction](https://www.mdpi.com/2076-3417/11/23/11514). Time aware Attention (TA) emphasizes data in the sequence by considering irregular time intervals. Feature similarity Attention (FA) estimates feature importance and similarities to emphasize data. Finally, the ensemble model of TA and FA-based BiLSTM can simultaneously represent both time and feature information. This repository only addresses the models with BiLSTM as the backbone. However, TA and FA as the attention modules can be easily applied in other backbones such as Transformer.
\
\
The overall architecture and attention process are as below:

<center><img src="./images/Model Process.jpg" width="70%" height="70%"></center>
<center><img src="./images/Attention Process.jpg" width="30%" height="30%"></center>

The models are designed to consider data properties, (i.e., sequential, irregular time, feature importance). The repository supports BiLSTM (BASE), Self-Attention (SA) with BiLSTM, TA with BiLSTM, FA with BiLSTM, and Ensemble. Notice that the repositoy does not support dataset.
 

# Installation & Enviornment

The OS, python and pytorch version needs as below:
- Windows / Linux 
- python >= 3.7.4
- pytorch >= 1.7.1
 
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


## Citation

If you use the code in your paper, then please cite it as:
```C
@article{park2021time,
  title={Time-Aware and Feature Similarity Self-Attention in Vessel Fuel Consumption Prediction},
  author={Park, Hyun Joon and Lee, Min Seok and Park, Dong Il and Han, Sung Won},
  journal={Applied Sciences},
  volume={11},
  number={23},
  pages={11514},
  year={2021},
  publisher={Multidisciplinary Digital Publishing Institute}
}
```
