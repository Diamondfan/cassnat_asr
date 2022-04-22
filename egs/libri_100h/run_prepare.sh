#!/usr/bin/env bash

# 2020 (Ruchao Fan)

# The data are already downloaded in the corresponding dir
data=/data/Databases/LibriSpeech/Librispeech

stage=3
end_stage=3
featdir=data/fbank

unit=char         #word piece
nbpe=1024
bpemode=unigram #bpe or unigram

. ./cmd.sh
. ./path.sh
. parse_options.sh

if [ $stage -le 1 ] && [ $end_stage -ge 1 ]; then
  # format the data as Kaldi data directories
  for part in dev-clean test-clean dev-other test-other train-clean-100; do
    # use underscore-separated names in data directories.
    local/data_prep.sh $data/$part data/$(echo $part | sed s/-/_/g)
  done
  echo "[Stage 1] Data Preparation Finished."
fi

train_set="train_clean_100"
train_set_sp="train_100h_sp"
test_set="dev_clean test_clean dev_other test_other"
dev_set="dev_clean dev_other"
if [ $stage -le 2 ] && [ $end_stage -ge 2 ]; then
    
  for part in $train_set; do
    steps/make_fbank.sh --nj 16 --cmd $cmd --write_utt2num_frames true \
      data/$part exp/make_fbank/$part $featdir/$part
  done

  # speed-perturbation
  utils/perturb_data_dir_speed.sh 0.9 data/train_clean_100 data/temp1
  utils/perturb_data_dir_speed.sh 1.0 data/train_clean_100 data/temp2
  utils/perturb_data_dir_speed.sh 1.1 data/train_clean_100 data/temp3
  utils/combine_data.sh --extra-files utt2uniq data/${train_set_sp} data/temp1 data/temp2 data/temp3
  rm -r data/temp1 data/temp2 data/temp3

  steps/make_fbank.sh --cmd $cmd --nj 16 --write_utt2num_frames true \
    data/${train_set_sp} exp/make_fbank/${train_set_sp} $featdir/$train_set_sp
  utils/fix_data_dir.sh data/${train_set_sp}
  
  #compute global cmvn with training data
  all_feats=data/train_feats.scp
  ( for f in $train_set_sp; do cat data/$f/feats.scp; done ) | sort -k1 > $all_feats

  #remember to replace cmvn.ark in training config and it is applied in dataloader
  compute-cmvn-stats scp:$all_feats data/fbank/cmvn_sp.ark || exit 1; 

  for part in $test_set; do
    steps/make_fbank.sh --nj 16 --cmd $cmd --write_utt2num_frames true \
      data/$part exp/make_fbank/$part $featdir/$part
  done

  echo "[Stage 2] Feature Extraction Finished"
fi

dict=data/dict/vocab_${unit}.txt ; mkdir -p data/dict
bpemodel=data/dict/bpemodel_${bpemode}_${nbpe}
if [ $stage -le 3 ] && [ $end_stage -ge 3 ]; then  
  echo "Create a dictionary..."
  all_text=data/train_text
  ( for f in $train_set_sp; do cat data/$f/text; done ) | sort -k1 > $all_text
  
  if [ $unit == wp ]; then
    cut -f 2- -d " " $all_text > data/dict/input.txt
    spm_train --input=data/dict/input.txt --vocab_size=$nbpe --model_type=$bpemode \
        --model_prefix=$bpemodel --input_sentence_size=100000000

    spm_encode --model=${bpemodel}.model --output_format=piece < data/dict/input.txt | tr ' ' '\n' | \
        sort | uniq | awk '{print $0 }' > $dict
  elif [ $unit == char ]; then
    for part in $train_set_sp $test_set; do
        python local/prepare_dict_char.py $dict $all_text data/$part/text > data/$part/token_char.scp
    done
  else
    echo "Not ImplementedError"; exit 1
  fi

  if [ $unit == wp ]; then
    for part in $train_set_sp $test_set; do
      paste -d " " <(awk '{print $1}' data/$part/text) <(cut -f 2- -d" " data/$part/text \
              | spm_encode --model=${bpemodel}.model --output_format=piece) \
              > data/$part/token_wp.scp
    done
  fi
  echo "[Stage 3] Dictionary and Transcription Finished."
fi



