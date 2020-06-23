#!/usr/bin/env bash

# 2020 (Ruchao Fan)

# The data are already downloaded in the corresponding dir
data=/home/ruchao/Database/LibriSpeech/
lm_data=/home/ruchao/Database/LibriSpeech/libri_lm

stage=5
end_stage=5
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
  all_text=data/train_text
  ( for f in $train_set; do cat data/$f/text; done ) | sort -k1 > $all_text
  
  if [ $unit == wp ]; then
    cut -f 2- -d " " $all_text > data/dict/input.txt
    spm_train --input=data/dict/input.txt --vocab_size=$nbpe --model_type=$bpemode \
        --model_prefix=$bpemodel --input_sentence_size=100000000

    spm_encode --model=${bpemodel}.model --output_format=piece < data/dict/input.txt | tr ' ' '\n' | \
        sort | uniq | awk '{print $0 }' > $dict
  else
    echo "Not ImplementedError"; exit 1
  fi

  for part in $train_set $test_set; do
    paste -d " " <(awk '{print $1}' data/$part/text) <(cut -f 2- -d" " data/$part/text \
            | spm_encode --model=${bpemodel}.model --output_format=piece) \
            > data/$part/token.scp
  done
  echo "[Stage 3] Dictionary and Transcription Finished."
fi

if [ $stage -le 4 ] && [ $end_stage -ge 4 ]; then
  #mkdir -p data/local
  #ln -s -r $lm_data data/local/lm

  echo "[Stage 4] External LM Training Finished."
fi

if [ $stage -le 5 ] && [ $end_stage -ge 5 ]; then
  exp=exp/transformer_960h_normal

  CUDA_VISIBLE_DEVICES="1" asr_train.py \
    --exp_dir $exp \
    --train_config conf/transformer.yaml \
    --data_config conf/data.yaml \
    --batch_size 32 \
    --epochs 15 \
    --save_epoch 10 \
    --anneal_lr_epoch 9 \
    --anneal_lr_ratio 0.5 \
    --learning_rate 0.0002 \
    --opt_type "normal" \
    --weight_decay 0.00001 \
    --label_smooth 0.1 \
    --ctc_alpha 0.3 \
    --print_freq 200 > $exp/train.log 2>&1 &
    

  echo "[Stage 5] ASR Training Finished."
fi

if [ $stage -le 6 ] && [ $end_stage -ge 6 ]; then
  # This is a very simple first version, only test on test_clean 
  exp=exp/transformer_960h
  
  CUDA_VISIBLE_DEVICES="1" test.py \
    --test_config conf/test.yaml \
    --data_path 'data/test_clean/feats.scp' \
    --resume_model $exp/test.mdl \
    --result_file $exp/test_clean_recog.scp \
    --batch_size 8 \
    --ctc_weight 0.3 \
    --rnnlm None \
    --lm_weight 0.0 \
    --max_decode_step 100 \
    --print_freq 100 
  
  text2trn.py $exp/test_clean_recog.scp $exp/test_clean_hyp.trn
  text2trn.py data/test_clean/token.scp $exp/test_clean_ref.trn
 
  spm_decode --model=${bpemodel}.model --input_format=piece < $exp/test_clean_hyp.trn | sed -e "s/▁/ /g" > $exp/hyp.wrd.trn
  spm_decode --model=${bpemodel}.model --input_format=piece < $exp/test_clean_ref.trn | sed -e "s/▁/ /g" > $exp/ref.wrd.trn
  sclite -r $exp/ref.wrd.trn -h $exp/hyp.wrd.trn -i rm -o all stdout > $exp/result.wrd.txt
  echo "[Stage 6] Decoding Finished."
fi


