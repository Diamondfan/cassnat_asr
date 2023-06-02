#!/usr/bin/env bash

# 2020 (Ruchao Fan)
# 2023 (Ruchao Fan) 
# SPAPL, UCLA

# The data are already downloaded in the corresponding dir
data=/data/Databases/Aishell1/

stage=1
end_stage=4
featdir=data/fbank

. ./cmd.sh
. ./path.sh
. parse_options.sh

if [ $stage -le 1 ] && [ $end_stage -ge 1 ]; then

  local/aishell_data_prep.sh ${data}/data_aishell/wav ${data}/data_aishell/transcript
  # remove space in text
  for x in train dev test; do
    cp data/${x}/text data/${x}/text.org
    paste -d " " <(cut -f 1 -d" " data/${x}/text.org) <(cut -f 2- -d" " data/${x}/text.org | tr -d " ") \
        > data/${x}/text
    rm data/${x}/text.org
  done
  echo "[Stage 1] Data Preparation Finished."
fi

train_set=train_all
dev_set=dev
if [ $stage -le 2 ] && [ $end_stage -ge 2 ]; then
  for part in train dev test; do
    steps/make_fbank.sh --nj 16 --cmd $cmd --write_utt2num_frames true \
      data/$part exp/make_fbank/$part $featdir/$part
    utils/fix_data_dir.sh data/$part
  done
  
  # speed-perturbed
  utils/perturb_data_dir_speed.sh 0.9 data/train data/temp1
  utils/perturb_data_dir_speed.sh 1.0 data/train data/temp2
  utils/perturb_data_dir_speed.sh 1.1 data/train data/temp3
  utils/combine_data.sh --extra-files utt2uniq data/${train_set} data/temp1 data/temp2 data/temp3
  rm -r data/temp1 data/temp2 data/temp3

  steps/make_fbank.sh --cmd $cmd --nj 32 --write_utt2num_frames true \
    data/${train_set} exp/make_fbank/${train_set} $featdir/$part
  utils/fix_data_dir.sh data/${train_set}

  # compute global CMVN
  compute-cmvn-stats scp:data/${train_set}/feats.scp data/fbank/cmvn.ark || exit 1;
  echo "[Stage 2] Feature Extraction Finished"
fi

dict=data/dict/vocab_char.txt ; mkdir -p data/dict
if [ $stage -le 3 ] && [ $end_stage -ge 3 ]; then  
  echo "Create a dictionary..."
  text2token.py -s 1 -n 1 data/${train_set}/text | cut -f 2- -d" " | tr " " "\n" \
    | sort | uniq | grep -v -e '^\s*$' | awk '{print $0}' > ${dict}

  for part in ${train_set} dev test; do
    text2token.py -s 1 -n 1 data/$part/text > data/$part/token.scp
  done
  echo "[Stage 3] Dictionary and Transcription Finished."
fi

if [ $stage -le 4 ] && [ $end_stage -ge 4 ]; then
  echo "Preparing Wav data for HuBERT encoder training."
  for sets in $train_set; do
    cut -d" " -f1 data/$sets/wav.scp | sed "s:\(.*\):\1 data/${sets}_wav/\1.wav:g" > data/$sets/wav_true.scp
    mkdir -p data/${sets}_wav/
    cat data/train_all/wav.scp  | sed "s:\(.*\) \(sox .*\) |:\2 > data/${sets}_wav/\1.wav:g" > a.sh
    bash a.sh && rm -f a.sh
  done
fi


