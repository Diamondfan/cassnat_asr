#!/usr/bin/env bash

# 2020 Ruchao Fan
# This script is to run our proposed CASS-NAT, The name is CASS-NAT
# (CTC alignement-based Signgle Step Non-autoregressive Transformer).

. cmd.sh
. path.sh

stage=3
end_stage=3
lm_model=exp/libri_tfunilm16x512_4card_cosineanneal_ep20_maxlen120/averaged.mdl
encoder_initial_model=exp/1kh_transformer_baseline_wotime_warp_f27t005/averaged.mdl
#asr_exp=exp/cassnat_multistep_initart_wosrc_wosrctrig_bimask/
#asr_exp=exp/cassnat_multistep_initart_wsrc_2blk_wsrctrig_bimask/
asr_exp=exp/cassnat_multistep_initart_wsrc_6blk_wosrctrig_bimask/

if [ $stage -le 1 ] && [ $end_stage -ge 1 ]; then

  if [ ! -d $asr_exp ]; then
    mkdir -p $asr_exp
  fi

  CUDA_VISIBLE_DEVICES="0,1,2,3" cassnat_train.py \
    --exp_dir $asr_exp \
    --train_config conf/cassnat_train.yaml \
    --data_config conf/data.yaml \
    --batch_size 16 \
    --epochs 100 \
    --save_epoch 40 \
    --end_patience 10 \
    --learning_rate 0.001 \
    --min_lr 0.00001 \
    --opt_type "multistep" \
    --weight_decay 0 \
    --label_smooth 0.1 \
    --ctc_alpha 1 \
    --interctc_alpha 0 \
    --interctc_layer 0 \
    --att_alpha 1 \
    --interce_alpha 0 \
    --interce_layer 0 \
    --use_cmvn \
    --init_encoder \
    --resume_model $encoder_initial_model \
    --seed 1234 \
    --print_freq 100 > $asr_exp/train.log 2>&1 &
    
  echo "[Stage 1] ASR Training Finished."
fi

out_name='averaged.mdl'
if [ $stage -le 2 ] && [ $end_stage -ge 2 ]; then
  last_epoch=81  # need to be modified
  
  average_checkpoints.py \
    --exp_dir $asr_exp \
    --out_name $out_name \
    --last_epoch $last_epoch \
    --num 12
  
  echo "[Stage 2] Average checkpoints Finished."

fi

#asr_exp=exp/conv_fanat_best_interctc05_ctc05_interce01_ce09_aftermapping/
#asr_exp=exp/conv_fanat_e10m2d4_max_specaug_multistep_initenc_convdec_maxlen8_kernel3_ctxtrig1
#asr_exp=exp_cassnat/fanat_large_specaug_multistep_trig_src_initenc_SchD_shift_path0
#asr_exp=exp/conv_fanat_best_interctc05_ctc05_interce01_ce09

if [ $stage -le 3 ] && [ $end_stage -ge 3 ]; then
  exp=$asr_exp

  bpemodel=data/dict/bpemodel_unigram_5000
  rank_model="lm" #"at_baseline" #"lm", "at_baseline"
  rnnlm_model=$lm_model
  #rnnlm_model=exp/1kh_transformer_baseline_wotime_warp_f27t005/averaged.mdl
  rank_yaml=conf/lm.yaml
  test_model=$asr_exp/$out_name
  decode_type='esa_att'
  attbm=1
  ctcbm=1 
  ctclm=0
  ctclp=0
  lmwt=0
  s_num=50
  threshold=0.9
  s_dist=0
  lp=0
  nj=16
  batch_size=1
  test_set="test_clean test_other dev_clean dev_other"

  for tset in $test_set; do
    echo "Decoding $tset..."
    desdir=$exp/${decode_type}_decode_attbm_${attbm}_sampdist_${s_dist}_samplenum_${s_num}_lm${lmwt}_threshold${threshold}_rank${rank_model}/$tset/

    if [ ! -d $desdir ]; then
      mkdir -p $desdir
    fi
    
    split_scps=
    for n in $(seq $nj); do
      split_scps="$split_scps $desdir/feats.$n.scp"
    done
    utils/split_scp.pl data/$tset/feats.scp $split_scps || exit 1;
    
    $cmd JOB=1:$nj $desdir/log/decode.JOB.log \
      CUDA_VISIBLE_DEVICES=JOB cassnat_decode.py \
        --test_config conf/cassnat_decode.yaml \
        --lm_config $rank_yaml \
        --rank_model $rank_model \
        --data_path $desdir/feats.JOB.scp \
        --text_label data/$tset/token.scp \
        --resume_model $test_model \
        --result_file $desdir/token_results.JOB.txt \
        --batch_size $batch_size \
        --decode_type $decode_type \
        --rnnlm $rnnlm_model \
        --lm_weight $lmwt \
        --max_decode_ratio 0 \
        --use_cmvn \
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
fi

