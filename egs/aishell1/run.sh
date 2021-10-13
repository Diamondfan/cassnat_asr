#!/usr/bin/env bash

# 2020 (Ruchao Fan)

# The data are already downloaded in the corresponding dir
data=/home/ruchao/Database/aishell/
lm_data=/home/ruchao/Database/LibriSpeech/libri_lm

stage=8
end_stage=8
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

exp=exp/ar_convenc_best_interctc05_ctc05/

if [ $stage -le 6 ] && [ $end_stage -ge 6 ]; then
  #exp=exp/ar_convenc_e6d6_d512_multistep30k_150k_ctc1_specaugt4m005_warp/

  if [ ! -d $exp ]; then
    mkdir -p $exp
  fi

  CUDA_VISIBLE_DEVICES="4,5,6,7" asr_train.py \
    --exp_dir $exp \
    --train_config conf/transformer.yaml \
    --data_config conf/data.yaml \
    --batch_size 32 \
    --epochs 100 \
    --save_epoch 40 \
    --anneal_lr_ratio 0.5 \
    --learning_rate 0.001 \
    --min_lr 0.00001 \
    --end_patience 10 \
    --opt_type "multistep" \
    --weight_decay 0 \
    --label_smooth 0.1 \
    --ctc_alpha 0.5 \
    --interctc_alpha 0.5 \
    --use_cmvn \
    --seed 1234 \
    --print_freq 50 > $exp/train.log 2>&1 &
    
  echo "[Stage 6] ASR Training Finished."
fi

out_name='averaged.mdl'
if [ $stage -le 7 ] && [ $end_stage -ge 7 ]; then
  #exp=exp/1kh_d512_multistep_ctc1_accum1_bth32_specaug
  last_epoch=69
  
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
  exp=exp/ar_convenc_e12d6_d256_multistep40k_160k_ctc1_accum1_bth32_specaugt2m40_warp/

  test_model=$exp/$out_name
  rnnlm_model=exp/averaged_lm.mdl
  global_cmvn=data/fbank/cmvn.ark
  decode_type='att_only'
  beam1=10 # check beam1 and beam2 in conf/decode.yaml, att beam
  beam2=10 #20 # ctc beam
  lp=0
  ctcwt=0.4
  lmwt=0 #0.7
  ctclm=0 #0.7
  ctclp=0 #2
  nj=1
  batch_size=1
  test_set="test" #dev test"

  for tset in $test_set; do
    echo "Decoding $tset..."
    desdir=$exp/${decode_type}_decode_average_ctc${ctcwt}_bm1_${beam1}_bm2_${beam2}_lmwt${lmwt}_ctclm${ctclm}_lp${lp}_ctclp${ctclp}_speech218_bth1_nj1/$tset/
    if [ ! -d $desdir ]; then
      mkdir -p $desdir
    fi
    
    split_scps=
    for n in $(seq $nj); do
      split_scps="$split_scps $desdir/feats.$n.scp"
    done
    utils/split_scp.pl data/$tset/feats.scp $split_scps || exit 1;
    
    $cmd JOB=1:$nj $desdir/log/decode.JOB.log \
      CUDA_VISIBLE_DEVICES=JOB asr_decode.py \
        --test_config conf/decode.yaml \
        --lm_config conf/lm.yaml \
        --data_path $desdir/feats.JOB.scp \
        --resume_model $test_model \
        --result_file $desdir/token_results.JOB.txt \
        --batch_size $batch_size \
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
 
    sclite -r $desdir/ref.token.trn -h $desdir/hyp.token.trn -i wsj -o all stdout > $desdir/result.wrd.txt
  done
  echo "[Stage 7] Decoding Finished."
fi


