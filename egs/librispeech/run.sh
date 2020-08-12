#!/usr/bin/env bash

# 2020 (Ruchao Fan)

# The data are already downloaded in the corresponding dir
data=/data/nas1/user/ruchao/Database/LibriSpeech/
lm_data=/data/nas1/user/ruchao/Database/LibriSpeech/libri_lm

stage=3
end_stage=3
featdir=data/fbank

unit=wp
nbpe=5000
bpemode=unigram #bpe or unigram

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
dev_set="dev_clean dev_other"
if [ $stage -le 2 ] && [ $end_stage -ge 2 ]; then
    
  for part in $train_set; do
    steps/make_fbank.sh --nj 32 --cmd $cmd --write_utt2num_frames true \
      data/$part exp/make_fbank/$part $featdir/$part
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
  echo "stage 4: LM Preparation"
  lmdir=data/lm_train
  if [ ! -d $lmdir ]; then
    mkdir -p $lmdir
  fi
  # use external data
  cat data/train_text | gzip -c > $lmdir/train_text.gz
  # combine external text and transcriptions and shuffle them with seed 777
  zcat $lm_data/librispeech-lm-norm.txt.gz $lmdir/train_text.gz |\
        spm_encode --model=${bpemodel}.model --output_format=piece > $lmdir/train.txt
  
  ( for f in $dev_set; do cat data/$f/text; done ) | sort -k1 | cut -f 2- -d" " |\
        spm_encode --model=${bpemodel}.model --output_format=piece > $lmdir/valid.txt

  echo "[Stage 4] LM Preparation Finished."
fi

if [ $stage -le 5 ] && [ $end_stage -ge 5 ]; then
  exp=exp/libri_tflm_unigram_4card_cosineanneal_ep10/
  if [ ! -d $exp ]; then
    mkdir -p $exp
  fi
  
  CUDA_VISIBLE_DEVICES="0,1,2,3" lm_train.py \
    --exp_dir $exp \
    --train_config conf/lm.yaml \
    --data_config conf/lm_data.yaml \
    --batch_size 64 \
    --epochs 10 \
    --save_epoch 3 \
    --anneal_lr_ratio 0.5 \
    --learning_rate 0.0001 \
    --min_lr 0.00001 \
    --patience 1 \
    --end_patience 5 \
    --opt_type "cosine" \
    --weight_decay 0 \
    --print_freq 200 > $exp/train.log 2>&1 &
 
  echo "[Stage 5] External LM Training Finished."
fi

if [ $stage -le 6 ] && [ $end_stage -ge 6 ]; then
  exp=exp/1kh_small_unigram_4card_ctc1_att1_multistep_accum1_gc5_spec_aug_first/

  if [ ! -d $exp ]; then
    mkdir -p $exp
  fi

  CUDA_VISIBLE_DEVICES="0,1,2,3" asr_train.py \
    --exp_dir $exp \
    --train_config conf/transformer.yaml \
    --data_config conf/data.yaml \
    --batch_size 32 \
    --epochs 100 \
    --save_epoch 40 \
    --anneal_lr_ratio 0.5 \
    --learning_rate 0.001 \
    --min_lr 0.00001 \
    --patience 1 \
    --end_patience 10 \
    --opt_type "multistep" \
    --weight_decay 0 \
    --label_smooth 0.1 \
    --ctc_alpha 1 \
    --use_cmvn \
    --print_freq 50 > $exp/train.log 2>&1 &
    
  echo "[Stage 6] ASR Training Finished."
fi

out_name='averaged.mdl'
if [ $stage -le 7 ] && [ $end_stage -ge 7 ]; then
  #exp=exp/1kh_small_unigram_4card_ctc1_att1_noamwarm_accum1_gc5_spec_aug/
  exp=exp/1kh_small_unigram_4card_ctc1_att1_multistep_accum1_gc5_spec_aug_first
  last_epoch=86  
  
  average_checkpoints.py \
    --exp_dir $exp \
    --out_name $out_name \
    --last_epoch $last_epoch \
    --num 12 
  
  #lm_exp=exp/libri_tflm_unigram_4card_cosineanneal_ep10/
  #last_epoch=9  
  
  #average_checkpoints.py \
  #  --exp_dir $lm_exp \
  #  --out_name $out_name \
  #  --last_epoch $last_epoch \
  #  --num 3

  echo "[Stage 7] Average checkpoints Finished."

fi

if [ $stage -le 8 ] && [ $end_stage -ge 8 ]; then
  exp=exp/1kh_small_unigram_4card_ctc1_att1_multistep_accum1_gc5_spec_aug_first/
  #exp=exp/1kh_small_unigram_4card_ctc1_att1_multistep_accum1_gc5_spec_aug

  test_model=$exp/$out_name
  rnnlm_model=exp/averaged_lm.mdl
  global_cmvn=data/fbank/cmvn.ark
  decode_type='ctc_att'
  beam1=10 # check beam1 and beam2 in conf/decode.yaml, att beam
  beam2=20 # ctc beam
  ctcwt=0.5
  lp=0
  lmwt=0.7
  nj=4
  
  for tset in $test_set; do
    echo "Decoding $tset..."
    desdir=$exp/${decode_type}_decode_average_ctc${ctcwt}_bm1_${beam1}_bm2_${beam2}_lmwt${lmwt}_lp${lp}/$tset/
    #desdir=$exp/decode_average_best/$tset
    if [ ! -d $desdir ]; then
      mkdir -p $desdir
    fi
    
    split_scps=
    for n in $(seq $nj); do
      split_scps="$split_scps $desdir/feats.$n.scp"
    done
    utils/split_scp.pl data/$tset/feats.scp $split_scps || exit 1;
    
    $cmd JOB=1:$nj $desdir/log/decode.JOB.log \
      CUDA_VISIBLE_DEVICES="0" asr_decode.py \
        --test_config conf/decode.yaml \
        --lm_config conf/lm.yaml \
        --data_path $desdir/feats.JOB.scp \
        --resume_model $test_model \
        --result_file $desdir/token_results.JOB.txt \
        --batch_size 8 \
        --decode_type $decode_type \
        --ctc_weight $ctcwt \
        --rnnlm $rnnlm_model \
        --lm_weight $lmwt \
        --max_decode_ratio 0 \
        --use_cmvn \
        --global_cmvn $global_cmvn \
        --print_freq 20 
    
    cat $desdir/token_results.*.txt | sort -k1,1 > $desdir/token_results.txt
    text2trn.py $desdir/token_results.txt $desdir/hyp.token.trn
    text2trn.py data/$tset/token.scp $desdir/ref.token.trn
 
    spm_decode --model=${bpemodel}.model --input_format=piece < $desdir/hyp.token.trn | sed -e "s/▁/ /g" |\
            sed -e "s/(/ (/g" > $desdir/hyp.wrd.trn
    spm_decode --model=${bpemodel}.model --input_format=piece < $desdir/ref.token.trn | sed -e "s/▁/ /g" |\
            sed -e "s/(/ (/g" > $desdir/ref.wrd.trn
    sclite -r $desdir/ref.wrd.trn -h $desdir/hyp.wrd.trn -i rm -o all stdout > $desdir/result.wrd.txt
  done
  echo "[Stage 7] Decoding Finished."
fi


