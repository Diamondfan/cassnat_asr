#!/usr/bin/env bash

# 2020 Ruchao Fan PAII Inc.
# 2022 Ruchao Fan SPAPL UCLA
# This script is to run our proposed CASS-NAT, The name is CASS-NAT
# (CTC alignement-based Signgle Step Non-autoregressive Transformer).


. cmd.sh
. path.sh

stage=3
end_stage=3
lm_model=exp/libri_tfunilm16x512_4card_cosineanneal_ep20_maxlen120/averaged.mdl

# cassnat settings
#train_config=conf/cassnat_train.yaml
#data_config=conf/data_raw.yaml
#start_saving_epoch=10

# cassnat with hubert encoder settings
train_config=conf/hubert_cassnat_train.yaml
data_config=conf/data_hubert.yaml
start_saving_epoch=5

#asr_exp=exp/100h_sptokenizer_cassnat_best_specaugt8_004/ #noam15k_initrand_interctc05_ly6_interce01_ly6_a4000_lr1e-3/ #mlm1p0/
asr_exp=exp/hubert_cassnat_lr5e-5_1e-3_samples_max80s_accum4_warmup10k_maskt05f05/

if [ $stage -le 1 ] && [ $end_stage -ge 1 ]; then

  [ ! -d $asr_exp ] && mkdir -p $asr_exp;

  CUDA_VISIBLE_DEVICES="0,1" train_asr.py \
    --task "cassnat" \
    --exp_dir $asr_exp \
    --train_config $train_config \
    --data_config $data_config \
    --optim_type "noam" \
    --epochs 60 \
    --start_saving_epoch $start_saving_epoch \
    --end_patience 10 \
    --seed 1234 \
    --print_freq 100 \
    --port 25142 > $asr_exp/train.log 2>&1 &
    
  echo "[Stage 1] ASR Training Finished."
fi

out_name='averaged.mdl'
if [ $stage -le 2 ] && [ $end_stage -ge 2 ]; then
  last_epoch=18  # need to be modified
  
  average_checkpoints.py \
    --exp_dir $asr_exp \
    --out_name $out_name \
    --last_epoch $last_epoch \
    --num 10
  
  echo "[Stage 2] Average checkpoints Finished."

fi

if [ $stage -le 3 ] && [ $end_stage -ge 3 ]; then
  exp=$asr_exp

  bpemodel=data/dict/bpemodel_unigram_1024
  rank_model="at_baseline" #"lm", "at_baseline"
  #rnnlm_model=$lm_model
  #rnnlm_model=exp/100h_sptokenizer_cfmer_interctc05_layer6_noam_warmup15k_lrpk1e-3_epoch60_2gpus/averaged.mdl
  #rank_yaml=conf/rank_model.yaml
  rnnlm_model=exp/hubert_art_lr5e-5_1e-3_samples_max80s_accum4_warmup10k_maskt06f05/averaged.mdl
  rank_yaml=conf/hubert_rank_model.yaml
  test_model=$asr_exp/$out_name
  decode_type='esa_att'
  attbm=1
  ctcbm=1 
  ctclm=0
  ctclp=0
  lmwt=0
  s_num=16
  threshold=0.9
  s_dist=0
  lp=0
  nj=1
  batch_size=1
  test_set="dev_clean dev_other" #test_clean test_other" # 
  # decode cassnat model
  #decode_config=conf/cassnat_decode.yaml
  #data_prefix=feats

  # decode cassnat model with hubert encoder
  decode_config=conf/hubert_cassnat_decode.yaml
  data_prefix=wav_s

  for tset in $test_set; do
    echo "Decoding $tset..."
    desdir=$exp/${decode_type}_decode_attbm_${attbm}_sampdist_${s_dist}_samplenum_${s_num}_lm${lmwt}_threshold${threshold}_rank${rank_model}/$tset/

    if [ ! -d $desdir ]; then
      mkdir -p $desdir
    fi
    
    split_scps=
    for n in $(seq $nj); do
      split_scps="$split_scps $desdir/${data_prefix}.$n.scp"
    done
    utils/split_scp.pl data/$tset/${data_prefix}.scp $split_scps || exit 1;
    
    $cmd JOB=1:$nj $desdir/log/decode.JOB.log \
      CUDA_VISIBLE_DEVICES="0" decode_asr.py \
        --task "cassnat" \
        --test_config $decode_config \
        --lm_config $rank_yaml \
        --rank_model $rank_model \
        --data_path $desdir/${data_prefix}.JOB.scp \
        --text_label data/$tset/text \
        --resume_model $test_model \
        --result_file $desdir/token_results.JOB.txt \
        --batch_size $batch_size \
        --rnnlm $rnnlm_model \
        --lm_weight $lmwt \
        --print_freq 20 
    
    cat $desdir/token_results.*.txt | sort -k1,1 > $desdir/token_results.txt
    text2trn.py $desdir/token_results.txt $desdir/hyp.token.trn
    text2trn.py data/$tset/token_wp.scp $desdir/ref.token.trn
 
    spm_decode --model=${bpemodel}.model --input_format=piece < $desdir/hyp.token.trn | sed -e "s/▁/ /g" |\
            sed -e "s/(/ (/g" > $desdir/hyp.wrd.trn
    spm_decode --model=${bpemodel}.model --input_format=piece < $desdir/ref.token.trn | sed -e "s/▁/ /g" |\
            sed -e "s/(/ (/g" > $desdir/ref.wrd.trn
    sclite -r $desdir/ref.wrd.trn -h $desdir/hyp.wrd.trn -i rm -o all stdout > $desdir/result.wrd.txt
  done
fi



