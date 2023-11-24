# Multi-target Extractor and Detector

The model for proposed in [Multi-Target Extractor and Detector for Unknown-Number Speaker Diarization](https://ieeexplore.ieee.org/document/10132569).

## Model
The model is the neural stage of MTEAD, stored in ``model/``
### Inputs
* X: Features, i.e. MFCC. Size: (Batch, Dim, Time)
* E: Speaker Features, i.e. 500 frames of speaker. Size: (Batch, Time , E_Dim, Spk)
* num_spks: Numbers of speaker, for training. Size: (Batch)
### Output
* Diarization (VAD) Result. Size: (Batch, Spk, Time)
### Model Structure
#### Main 

<img src="https://i.imgur.com/683uyaA.png" width="500" height="500">

#### Time-speaker Contextualizer

<img src="https://i.imgur.com/YAb1Mma.png" width="450" height="300">

### Dimension of each layer

```bash
=========================================================================================================
Layer (type:depth-idx)                                  Output Shape              Param #
=========================================================================================================
Model                                                   [32, 4, 500]              --
├─Linear: 1-1                                           [128, 500, 513]           21,033
├─ResNet: 1-2                                           [128, 256]                --
│    └─Sequential: 2-1                                  [128, 128, 500]           --
│    │    └─Conv1d: 3-1                                 [128, 1024, 500]          1,577,984
│    │    └─Conv1d_Layernorm_LRelu_Residual: 3-2        [128, 1024, 500]          6,299,648
│    │    └─Conv1d_Layernorm_LRelu_Residual: 3-3        [128, 1024, 500]          6,299,648
│    │    └─Conv1d_Layernorm_LRelu_Residual: 3-4        [128, 1024, 500]          6,299,648
│    │    └─LeakyReLU: 3-5                              [128, 1024, 500]          --
│    │    └─Conv1d: 3-6                                 [128, 512, 500]           1,573,888
│    │    └─Conv1d_Layernorm_LRelu_Residual: 3-7        [128, 512, 500]           1,576,960
│    │    └─Conv1d_Layernorm_LRelu_Residual: 3-8        [128, 512, 500]           1,576,960
│    │    └─Conv1d_Layernorm_LRelu_Residual: 3-9        [128, 512, 500]           1,576,960
│    │    └─LeakyReLU: 3-10                             [128, 512, 500]           --
│    │    └─Conv1d: 3-11                                [128, 256, 500]           393,728
│    │    └─Conv1d_Layernorm_LRelu_Residual: 3-12       [128, 256, 500]           395,264
│    │    └─Conv1d_Layernorm_LRelu_Residual: 3-13       [128, 256, 500]           395,264
│    │    └─Conv1d_Layernorm_LRelu_Residual: 3-14       [128, 256, 500]           395,264
│    │    └─LeakyReLU: 3-15                             [128, 256, 500]           --
│    │    └─Conv1d: 3-16                                [128, 128, 500]           98,560
│    │    └─Conv1d_Layernorm_LRelu_Residual: 3-17       [128, 128, 500]           99,328
│    │    └─Conv1d_Layernorm_LRelu_Residual: 3-18       [128, 128, 500]           99,328
│    │    └─Conv1d_Layernorm_LRelu_Residual: 3-19       [128, 128, 500]           99,328
│    │    └─LeakyReLU: 3-20                             [128, 128, 500]           --
│    └─AttentiveStatisticPooling: 2-2                   [128, 256]                --
│    │    └─BatchNorm1d: 3-21                           [128, 128, 500]           256
│    │    └─Conv1d: 3-22                                [128, 128, 500]           16,640
│    │    └─Conv1d: 3-23                                [128, 1, 500]             129
│    │    └─Softmax: 3-24                               [128, 1, 500]             --
├─Linear: 1-3                                           [32, 4, 128]              32,896
├─CNN: 1-4                                              [32, 128, 500]            --
│    └─Sequential: 2-3                                  [32, 128, 500]            --
│    │    └─Conv1d: 3-25                                [32, 128, 500]            15,488
│    │    └─ReLU: 3-26                                  [32, 128, 500]            --
│    │    └─BatchNorm1d: 3-27                           [32, 128, 500]            256
│    │    └─Conv1d: 3-28                                [32, 128, 500]            49,280
│    │    └─ReLU: 3-29                                  [32, 128, 500]            --
│    │    └─BatchNorm1d: 3-30                           [32, 128, 500]            256
│    │    └─Conv1d: 3-31                                [32, 128, 500]            49,280
│    │    └─ReLU: 3-32                                  [32, 128, 500]            --
│    │    └─BatchNorm1d: 3-33                           [32, 128, 500]            256
│    │    └─Conv1d: 3-34                                [32, 128, 500]            49,280
│    │    └─ReLU: 3-35                                  [32, 128, 500]            --
│    │    └─BatchNorm1d: 3-36                           [32, 128, 500]            256
├─RNN: 1-5                                              [128, 128, 500]           --
│    └─LSTM: 2-4                                        [128, 500, 128]           264,192
│    └─Linear: 2-5                                      [128, 500, 128]           16,512
├─TimeSpeakerContextualizer: 1-6                        [32, 4, 500]              --
│    └─Sequential: 2-6                                  [32, 128, 4, 500]         --
│    │    └─GroupNorm: 3-37                             [32, 128, 4, 500]         256
│    │    └─Conv2d: 3-38                                [32, 128, 4, 500]         16,512
│    └─ModuleList: 2-11                                 --                        (recursive)
│    │    └─LSTMBlock: 3-39                             [32, 500, 4, 128]         116,096
│    └─ModuleList: 2-12                                 --                        (recursive)
│    │    └─LSTMBlock: 3-40                             [32, 4, 500, 128]         116,096
│    └─ModuleList: 2-11                                 --                        (recursive)
│    │    └─LSTMBlock: 3-41                             [32, 500, 4, 128]         116,096
│    └─ModuleList: 2-12                                 --                        (recursive)
│    │    └─LSTMBlock: 3-42                             [32, 4, 500, 128]         116,096
│    └─ModuleList: 2-11                                 --                        (recursive)
│    │    └─LSTMBlock: 3-43                             [32, 500, 4, 128]         116,096
│    └─ModuleList: 2-12                                 --                        (recursive)
│    │    └─LSTMBlock: 3-44                             [32, 4, 500, 128]         116,096
│    └─Sequential: 2-13                                 [32, 128, 500, 4]         --
│    │    └─PReLU: 3-45                                 [32, 128, 500, 4]         1
│    │    └─Conv2d: 3-46                                [32, 128, 500, 4]         16,512
│    └─Sequential: 2-14                                 [128, 1, 500]             --
│    │    └─Conv1d: 3-47                                [128, 256, 500]           33,024
│    │    └─GroupNorm: 3-48                             [128, 256, 500]           512
│    │    └─GatedTanhUnit: 3-49                         [128, 128, 500]           --
│    │    └─Conv1d: 3-50                                [128, 1, 500]             128
=========================================================================================================
Total params: 30,037,291
Trainable params: 30,037,291
Non-trainable params: 0
Total mult-adds (T): 4.97
=========================================================================================================
Input size (MB): 12.80
Forward/backward pass size (MB): 15086.72
Params size (MB): 120.15
Estimated Total Size (MB): 15219.67
=========================================================================================================

```

## Configuration
* Optimizer: Adam
    * learning rate: 0.1
    * max gradiant norm: 5
* LR scheduler: noam
    * warmup steps: 25000
* See ``conf/`` for more information

## Requirements
 - python >= 3.6
   - pytorch >= 1.3
  
## Citation
```
@ARTICLE{10132569,
  author={Cheng, Chin-Yi and Lee, Hung-Shin and Tsao, Yu and Wang, Hsin-Min},
  journal={IEEE Signal Processing Letters}, 
  title={Multi-Target Extractor and Detector for Unknown-Number Speaker Diarization}, 
  year={2023},
  volume={30},
  number={},
  pages={638-642},
  doi={10.1109/LSP.2023.3279781}
}
```

