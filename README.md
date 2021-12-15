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

You can install requirements through git and requirements.txt except for pytorch and torchaudio.
```C
git clone https://github.com/winddori2002/Time-aware-and-Feature-similarity-Self-Attention.git
cd Time-aware-and-Feature-similarity-Self-Attention
pip install -r requirements.txt
```

# Usage

Since the repository does not support datasets, it is necessary to fit your data into the code.
Other numeric features can be used easily, but you need to get the ```"time diff"``` feature, which is the
time difference right before data in the sequence. The unit of time diff in the paper is a minute.
Otherwise, you can use the part of the model and attention modules.

## 1. Train

### Training with default settings

You can train the model with default setting by running the following code.

```C
python main.py train --att_type [attention type]
```

### Training with other arguments
If you want to edit model settings, you can run the following code with other arguments. 

In ```config.py```, you can find other arguments, such as batch size, epoch, and so on.

```
python main.py train --hidden 60 --depth 4 --growth 2 --kernel_size 8 --stride 4 --segment_len 64 --aug True --aug_type tempo

MANNER arguments:
  --in_channels: initial in channel size (default:1)
  --out_channels: initial out channel size (default:1)
  --hidden: channel size to expand (default:60)
  --depth: number of layers for encoder and decoder (default:4)
  --kernel_size: kernel size for UP/DOWN conv (default:8)
  --stride: stride for UP/DOWN conv (default:4)
  --growth: channel expansion ration (default:2)
  --head: number of head for global attention (default:1)
  --segment_len: chunk size for overlapped chunking (default:64)
  
Setting arguments:
  --sample_rate: sample_rate (default:16000)
  --segment: segment the audio signal with seconds (default:4)
  --set_stride: Overlapped seconds when segment the signal (default:1)
  
Augmentation arguments:
  --aug: True/False 
  --aug_type: augmentation type (tempo, speed, shift available. only shift available on Windows.)
```


## 2. Test

```C
python main.py test --att_type [attention type]
```

## 3. parameter settings
### Possible to change parameters including paths, when you call main.py.

* epoch: 100
* batch_size: 200
* lr: 0.001

* att_type: BASE / SA / TA / FA / ENS
* hidden_size: 4
* sequence_length: 50
* num_layers: 2
* dropout: 0.1
* function_type: sig


# Visualizations

## 1. Feature importance
### Estimated feature importance from FA model. Feature importance is learnable parameters.

<center><img src="./images/Importance.jpg" width="40%" height="40%"></center>

## 2. TA attention map
### One example of TA attention map. TA ignores point where time difference increases rapidly.

<center><img src="./images/TA_MAP.jpg" width="40%" height="40%"></center>


## 3. FA attention map
### One example of FA attention map. FA makes symmetric attention maps and FA is especially affected by Speed.

<center><img src="./images/FA_MAP.jpg" width="40%" height="40%"></center>


# Citation

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
