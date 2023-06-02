#!/usr/bin/env bash

# 2020 (Ruchao Fan)
# 2023 (Ruchao Fan) 
# SPAPL, UCLA

stage=1
end_stage=1
featdir=data/fbank

. ./cmd.sh
. ./path.sh
. parse_options.sh

# art training
#train_config=conf/art_train.yaml
#data_config=conf/data_raw.yaml
#start_saving_epoch=30

# art with hubert encoder training
train_config=conf/hubert_art_train.yaml
data_config=conf/data_hubert.yaml
start_saving_epoch=1

#asr_exp=exp/ar_conformer_baseline_interctc05_layer6_spect10m005f2m27_multistep1k30k120k/
asr_exp=exp/hubert_ar_conformer_maskt05f05_multistep1k30k120k/

if [ $stage -le 1 ] && [ $end_stage -ge 1 ]; then
  [ ! -d $asr_exp ] && mkdir -p $asr_exp

  CUDA_VISIBLE_DEVICES="2,3" train_asr.py \
    --task "art" \
    --exp_dir $asr_exp \
    --train_config $train_config \
    --data_config $data_config \
    --optim_type "multistep" \
    --epochs 50 \
    --start_saving_epoch $start_saving_epoch \
    --end_patience 10 \
    --seed 1234 \
    --print_freq 100 \
    --port 21526 >> $asr_exp/train.log 2>&1 &
    
  echo "[Stage 1] ASR Training Finished."
fi

out_name='averaged.mdl'
if [ $stage -le 2 ] && [ $end_stage -ge 2 ]; then
  last_epoch=30
  
  average_checkpoints.py \
    --exp_dir $asr_exp \
    --out_name $out_name \
    --last_epoch $last_epoch \
    --num 10

  echo "[Stage 2] Average checkpoints Finished."

fi

if [ $stage -le 3 ] && [ $end_stage -ge 3 ]; then
  exp=$asr_exp
  lm_exp=
  test_model=$exp/$out_name
  rnnlm_model=$lm_exp/averaged_lm.mdl
  decode_type='ctc_att'
  beam1=10 # check beam1 and beam2 in conf/decode.yaml, att beam
  beam2=10 #20 # ctc beam
  lp=0
  ctcwt=0.4
  lmwt=0
  ctclm=0
  ctclp=0
  nj=1
  batch_size=1
  test_set="dev test"

  # decode art model
  #decode_config=conf/art_decode.yaml
  #data_prefix=feats

  # decode art model with hubert encoder
  decode_config=conf/hubert_art_decode.yaml
  data_prefix=wav_s

  for tset in $test_set; do
    echo "Decoding $tset..."
    desdir=$exp/${decode_type}_decode_average_ctc${ctcwt}_bm1_${beam1}_bm2_${beam2}_lmwt${lmwt}_ctclm${ctclm}_lp${lp}_ctclp${ctclp}_bth1_nj1/$tset/
    if [ ! -d $desdir ]; then
      mkdir -p $desdir
    fi
    
    split_scps=
    for n in $(seq $nj); do
      split_scps="$split_scps $desdir/${data_prefix}.$n.scp"
    done
    utils/split_scp.pl data/$tset/${data_prefix}.scp $split_scps || exit 1;
    
    $cmd JOB=1:$nj $desdir/log/decode.JOB.log \
      CUDA_VISIBLE_DEVICES="2" decode_asr.py \
        --task "art" \
        --test_config $decode_config \
        --lm_config conf/lm.yaml \
        --data_path $desdir/${data_prefix}.JOB.scp \
        --resume_model $test_model \
        --result_file $desdir/token_results.JOB.txt \
        --batch_size $batch_size \
        --rnnlm $rnnlm_model \
        --lm_weight $lmwt \
        --print_freq 20 
    
    cat $desdir/token_results.*.txt | sort -k1,1 > $desdir/token_results.txt
    text2trn.py $desdir/token_results.txt $desdir/hyp.token.trn
    text2trn.py data/$tset/token.scp $desdir/ref.token.trn
 
    sclite -r $desdir/ref.token.trn -h $desdir/hyp.token.trn -i wsj -o all stdout > $desdir/result.wrd.txt
  done
  echo "[Stage 7] Decoding Finished."
fi


