# Speech-transformer (Auto-regressive and Non-autoregressive)

This is the implementation of our work "Using CTC alignments as latent variables for Non-autoregressive speech-transformer". Some codes are borrowed from [Espnet](https://github.com/espnet/espnet) and [transformer implementation in Harvard NLP group](https://nlp.seas.harvard.edu/2018/04/03/attention.html).

## 1. Requirements

- Python 3.7
- Pytorch 1.2
- Kaldi

We didn't test it for a higher version of Python or Pytorch. Other required python packages are in requirments.txt. You can install it using:
```
pip install -r requirements.txt
```

## 2. Example, run librispeech.

1. Go to egs/librispeech. Modify path.sh and specify the kaldi path (for feature extraction and etc.).
2. Check the conf/transformer.yaml and make revisions on hyparameters if you like.
3. ./run.sh. I suggest to run the script step by step.
4. ./run\_fanat.sh. Run the non-autoregressive model. You can directly run this step if you want to skip the Auto-regressive transformer.

All the python codes are under src/. Some codes may not well organized since this is still in the period of experiments

## 3. Results.

- Librispeech (WER)

| Methods |  LM  | dev-clean | test-clean | dev-other | test-other | RTF(s) |
|   :-:   |  :-: |    :-:    |     :-:    |    :-:    |    :-:     | :-:    |
|   AST   |  no  |    3.4    |     3.6    |    8.5    |    8.5     | 0.562  |
|   -     |  yes |    2.5    |     2.7    |    5.7    |    5.8     |   -    |
|   NAST  |  no  |    3.7    |     3.8    |    9.2    |    9.1     | 0.011  |
|   -     |  yes |    3.3    |     3.3    |    8.0    |    8.1     |   -    |

- Aishell1 (CER)

| Methods |  LM  | dev  | test  | 
|   :-:   |  :-: | :-:  | :-:   | 
|   AST   |  no  | 5.4  |  5.9  |
|   NAST  |  no  | 5.3  |  5.8  |




