#!/usr/bin/env bash

# 2021 (Ruchao Fan)

data=/data/Databases/MyST/myst-v0.4.2/

stage=3
end_stage=3
featdir=data/fbank

unit=char #wp         #character
nbpe=500
bpemode=unigram #bpe or unigram

. ./cmd.sh
. ./path.sh
. parse_options.sh

if [ $stage -le 1 ] && [ $end_stage -ge 1 ]; then
  # format the data as Kaldi data directories
  for x in train development test; do
    local/myst_data_prepare.sh $data/data/$x data/$x
  done

  for x in train development test; do
    cat data/$x/text | awk -F' ' '{if(NF>1) {print $0}}' | cut -d" " -f1  > utt_list
    utils/subset_data_dir.sh --utt-list utt_list data/$x data/tmp
    rm -r data/$x
    mv data/tmp data/$x
  done
  rm utt_list

  echo "[Stage 1] Data Preparation Finished."
fi

train_set="train_sp"
test_set="development test"
dev_set="development"
if [ $stage -le 2 ] && [ $end_stage -ge 2 ]; then

  steps/make_fbank.sh --nj 8 --cmd $cmd --write_utt2num_frames true \
    data/train exp/make_fbank/$part $featdir/train

  utils/perturb_data_dir_speed.sh 0.9  data/train  data/temp1
  utils/perturb_data_dir_speed.sh 1.0  data/train  data/temp2
  utils/perturb_data_dir_speed.sh 1.1  data/train  data/temp3

  utils/combine_data.sh data/train_sp data/temp1 data/temp2 data/temp3  
  rm -r data/temp*
   
  for part in train_sp; do
    steps/make_fbank.sh --nj 8 --cmd $cmd --write_utt2num_frames true \
      data/$part exp/make_fbank/$part $featdir/$part
  done

  #compute global cmvn with training data
  all_feats=data/train_feats.scp
  ( for f in train_sp; do cat data/$f/feats.scp; done ) | sort -k1 > $all_feats

  #remember to replace cmvn.ark in training config and it is applied in dataloader
  compute-cmvn-stats scp:$all_feats data/fbank/cmvn_sp.ark || exit 1;

  for part in $test_set; do
    steps/make_fbank.sh --nj 8 --cmd $cmd --write_utt2num_frames true \
      data/$part exp/make_fbank/$part $featdir/$part
  done

  echo "[Stage 2] Feature Extraction Finished"
fi

dict=data/dict/vocab_${unit}.txt; mkdir -p data/dict
bpemodel=data/dict/bpemodel_${bpemode}_${nbpe}
if [ $stage -le 3 ] && [ $end_stage -ge 3 ]; then
  echo "Create a dictionary..."
  all_text=data/train_text
  ( for f in $train_set; do cat data/$f/text; done ) | sort -k1 > $all_text

  if [ $unit == wp ]; then
    cut -f 2- -d " " $all_text > data/dict/input.txt
    spm_train --input=data/dict/input.txt --vocab_size=$nbpe --model_type=$bpemode \
        --model_prefix=$bpemodel --input_sentence_size=100000

    spm_encode --model=${bpemodel}.model --output_format=piece < data/dict/input.txt | tr ' ' '\n' | \
        sort | uniq | awk '{print $0 }' > $dict

    for part in $train_set $test_set; do
      paste -d " " <(awk '{print $1}' data/$part/text) <(cut -f 2- -d" " data/$part/text \
            | spm_encode --model=${bpemodel}.model --output_format=piece) \
            > data/$part/token_wp.scp
    done
  elif [ $unit == char ]; then
    for part in $train_set $test_set; do
        python local/prepare_dict_char.py $dict $all_text data/$part/text > data/$part/token_char.scp
    done
  else
    echo "Not ImplementedError"; exit 1
  fi

  echo "[Stage 3] Dictionary and Transcription Finished."
fi
