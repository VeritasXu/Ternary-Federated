# Ternary-Federated
> Code, requirements and acknowledgements.

This is the source code for the paper "Xu J, Du W, Jin Y, et al. Ternary compression for communication-efficient federated learning[J]. IEEE Transactions on Neural Networks and Learning Systems, 2020.",  [Link](https://ieeexplore.ieee.org/abstract/document/9288933)


## Requirements

```
#requirements
Pytorch 1.0.0 or higher
torchvision 0.6.0 or higher
```



## Running

Terminal / cmd in Linux / Windows

```
cd Ternary-Federated
python Ternary_Fed.py
```



## Results

We use the moderate CNN in the paper to examine the performance of the proposed method a second time, please note that the "up+down" means that the Strategy II in the Algorithm 2 of the paper is not used in this experiment, and we have not save the best model during the training, hence, the oscillation will be more obvious. In practice, you can choose not to save the best model in previous rounds, or you can set a tiny gap for Strategy II (from 3% to 1% for example) for a steady curve.


<p align="center">
  <img src="https://github.com/VeritasXu/Ternary-Federated/blob/master/Results/moderate_cnn.png" alt="demo1" style="zoom:10%;"/>
</p>



|    Methods    | FedAvg | T-FedAvg (upstream reduced) | T-FedAvg (up&downstream reduced) |
| :-----------: | :----: | :-------------------------: | :------------------------------: |
| Best accuracy | 90.98% |           90.77%            |              90.55%              |



## Motivations

- Communication efficient federated learning
  - [x] upstream 
  - [x] downstream
- Robust for local epochs: can achieve satisfied performance with limited communications without setting local epoch to 1
- 
- Theoretical analysis

## Drawbacks

- quantization errors
- running efficiency



## Acknowledgements

The motivations are inspired by  [Trained ternary quantization](https://arxiv.org/pdf/1612.01064.pdf "optional title") and [Ternary weight networks](https://arxiv.org/pdf/1605.04711.pdf "optional title"), and the implementation of the code is highly relied on [PyTorch TTQ](https://github.com/TropComplique/trained-ternary-quantization "optional title"), many thanks for the efforts of the authors. We are also grateful for the open-source spirit of the machine / deep learning communities. 



## Note

- [x] We can supply another version of code with local area network (LAN) communications based on **flask and PyTorch**, please contact us if necessary.

- [x] Distributed under the MIT license. See ``LICENSE`` for more information.
- [ ] This work has certain disadvantages due to my limited ability, if you have any suggestions or comments, thank you very much for letting me know.

## Citation

GB/T 7714

```
Xu J, Du W, Jin Y, et al. Ternary compression for communication-efficient federated learning[J]. IEEE Transactions on Neural Networks and Learning Systems, 2020.
```

Bibtex

```tex
@article{xu2020ternary,
  title={Ternary compression for communication-efficient federated learning},
  author={Xu, Jinjin and Du, Wenli and Jin, Yaochu and He, Wangli and Cheng, Ran},
  journal={IEEE Transactions on Neural Networks and Learning Systems},
  year={2020},
  publisher={IEEE}
}
```

