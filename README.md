# Auto-regressive and Non-autoregressive Transformer for Speech Recognition

This is the implementation of our work "CTC alignment-based Non-autoregressive Speech Transformer". Some codes are borrowed from [Espnet](https://github.com/espnet/espnet) and [transformer implementation in Harvard NLP group](https://nlp.seas.harvard.edu/2018/04/03/attention.html).

## News:
- Using pretrained Hubert Encoder for CASS-NAT.

## Requirements

- Python 3.7
- Pytorch 1.11
- Kaldi

We didn't test it for a higher version of Python or Pytorch. Other required python packages are in requirments.txt. You can install it using:
```
pip install -r requirements.txt
```

## Example, run librispeech (scripts under libri_100 are tested).

1. Go to egs/librispeech. Modify path.sh and specify the kaldi path (for feature extraction and etc.).
2. Check the conf/transformer.yaml and make revisions on hyparameters if you like.
3. ./run.sh. I suggest to run the script step by step.
4. ./run\_cassnat.sh. Run the non-autoregressive model. You can directly run this step if you want to skip the Auto-regressive transformer.

All the python codes are under src/. Some codes may not well organized since this is still in the period of experiments

## Results (need updates for conformer encoder and hubert encoder).

- Librispeech (WER)

| Methods |  LM  | dev-clean | test-clean | dev-other | test-other | RTF(s) |
|   :-:   |  :-: |    :-:    |     :-:    |    :-:    |    :-:     | :-:    |
|   AT    |  no  |    3.4    |     3.6    |    8.5    |    8.5     | 0.562  |
|   -     |  yes |    2.5    |     2.7    |    5.7    |    5.8     |   -    |
| ConAT   |  no  |    2.7    |     3.0    |    7.2    |    7.0     | 0.499  |
| CASSNAT |  no  |    3.7    |     3.8    |    9.2    |    9.1     | 0.011  |
|   -     |  yes |    3.3    |     3.3    |    8.0    |    8.1     |   -    |
| ImpCASS |  no  |    2.8    |     3.1    |    7.3    |    7.2     | 0.014  |

- Aishell1 (CER)

| Methods |  LM  | dev  | test  | 
|   :-:   |  :-: | :-:  | :-:   | 
|   AT    |  no  | 5.4  |  5.9  |
| CASSNAT |  no  | 5.3  |  5.8  |
| ImpCASS |  no  | 4.9  |  5.4  |

## Citations

If you find this repository useful, please consider citing our work:

```
@inproceedings{cassnat,
  title={Cass-nat: Ctc alignment-based single step non-autoregressive transformer for speech recognition},
  author={Fan, Ruchao and Chu, Wei and Chang, Peng and Xiao, Jing},
  booktitle={IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={5889--5893},
  year={2021},
  organization={IEEE}
}
```

```
@inproceedings{improvedcassnat,
  author={Ruchao Fan and Wei Chu and Peng Chang and Jing Xiao and Abeer Alwan},
  title={{An Improved Single Step Non-Autoregressive Transformer for Automatic Speech Recognition}},
  year=2021,
  booktitle={Proc. Interspeech 2021},
  pages={3715--3719},
  doi={10.21437/Interspeech.2021-1955}
}
```

```
@article{studycassnat,
  author    = {Ruchao Fan and Wei Chu and Peng Chang and Abeer Alwan},
  title     = {A {CTC} Alignment-based Non-autoregressive Transformer for End-to-end Automatic Speech Recognition},
  journal   = {IEEE Transactions on Audio, Speech and Language Processing},
  doi       = {10.1109/TASLP.2023.3263789},
  year      = {2023}
}
```


