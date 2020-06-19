#!/usr/bin/env bash

# 2020 (Ruchao Fan)

# The data are already downloaded in the corresponding dir
data=/home/ruchao/Database/LibriSpeech/
lm_data=/home/ruchao/Database/LibriSpeech/libri_lm

stage=4
end_stage=10
featdir=data/fbank

unit=wp
nbpe=5000
bpemode=bpe #bpe or unigram

. ./cmd.sh
. ./path.sh
. parse_options.sh

if [ $stage -le 1 ] && [ $end_stage -ge 1 ]; then
  # format the data as Kaldi data directories
  for part in dev-clean test-clean dev-other test-other train-clean-100 train-clean-360 train-other-500; do
    # use underscore-separated names in data directories.
    local/data_prep.sh $data/LibriSpeech/$part data/$(echo $part | sed s/-/_/g)
  done
  echo "[Stage 1] Data Preparation Finished."
fi

train_set="train_clean_100 train_clean_360 train_other_500"
test_set="dev_clean test_clean dev_other test_other"
if [ $stage -le 2 ] && [ $end_stage -ge 2 ]; then
    
  for part in $train_set; do
    steps/make_fbank.sh --nj 32 --cmd $cmd --write_utt2num_frames true \
      data/$part exp/make_fbank/$part $featdir/$part
    cat data/$part/feats.scp >> data/train_feats.scp
  done
  
  #compute global cmvn with training data
  all_feats=data/train_feats.scp
  ( for f in $train_set; do cat data/$f/feats.scp; done ) | sort -k1 > $all_feats

  #remember to replace cmvn.ark in training config and it is applied in dataloader
  compute-cmvn-stats scp:$all_feats data/fbank/cmvn.ark || exit 1; 

  for part in $test_set; do
    steps/make_fbank.sh --nj 32 --cmd $cmd --write_utt2num_frames true \
      data/$part exp/make_fbank/$part $featdir/$part
  done

  echo "[Stage 2] Feature Extraction Finished"
fi

dict=data/dict/vocab_${unit}.txt ; mkdir -p data/dict
bpemodel=data/dict/bpemodel_${bpemode}_${nbpe}
if [ $stage -le 3 ] && [ $end_stage -ge 3 ]; then  
  echo "Create a dictionary..."
  echo "<UNK> 1" > $dict
  all_text=data/train_text
  ( for f in $train_set; do cat data/$f/text; done ) | sort -k1 > $all_text
  
  if [ $unit == wp ]; then
    cut -f 2- -d " " $all_text > data/dict/input.txt
    spm_train --input=data/dict/input.txt --vocab_size=$nbpe --model_type=$bpemode \
        --model_prefix=$bpemodel --input_sentence_size=100000000

    spm_encode --model=${bpemodel}.model --output_format=piece < data/dict/input.txt | tr ' ' '\n' | \
        sort | uniq | awk '{print $0" "NR+1 }' >> $dict
  else
    echo "Not ImplementedError"; exit 1
  fi

  for part in $train_set $test_set; do
    paste -d " " <(awk '{print $1}' data/$part/text) <(cut -f 2- -d" " data/$part/text \
            | spm_encode --model=${bpemodel}.model --output_format=piece) \
            > data/$part/token.scp
    cat data/$part/token.scp | utils/sym2int.pl --map-oov "<UNK>" -f 2- $dict > data/$part/tokenid.scp
  done
  echo "[Stage 3] Dictionary and Transcription Finished."
fi

if [ $stage -le 4 ] && [ $end_stage -ge 4 ]; then
  #mkdir -p data/local
  #ln -s -r $lm_data data/local/lm

  echo "[Stage 4] External LM Training Finished."
fi

if [ $stage -le 5 ] && [ $end_stage -ge 5 ]; then

  CUDA_VISIBLE_DEVICES="0" asr_train.py \
    --data_config conf/data.yaml \
    --config \
  echo "[Stage 5] ASR Training Finished."
fi

if [ $stage -le 6 ] && [ $end_stage -ge 6 ]; then

  echo "[Stage 6] Decoding Finished."
fi


