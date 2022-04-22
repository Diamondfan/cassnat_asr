#!/usr/bin/env bash

# 2020 (Ruchao Fan)
# 2022

stage=2
end_stage=2
featdir=data/fbank

unit=wp         #word piece
nbpe=1024
bpemode=unigram #bpe or unigram

. ./cmd.sh
. ./path.sh
. parse_options.sh

asr_exp=exp/100h_wp_tsfm_noam_warmup25k_lrpk1e-3_epoch80_2gpus/
#asr_exp=exp/100h_wp_tsfm_multistep2k50k200k_lrpk1e-3_epoch100_2gpus/

if [ $stage -le 1 ] && [ $end_stage -ge 1 ]; then

  [ ! -d $asr_exp ] && mkdir -p $asr_exp

  CUDA_VISIBLE_DEVICES="0,1" train_asr.py \
    --task "art" \
    --exp_dir $asr_exp \
    --train_config conf/transformer.yaml \
    --data_config conf/data_wp.yaml \
    --optim_type "multistep" \
    --epochs 100 \
    --start_saving_epoch 50 \
    --end_patience 10 \
    --seed 1234 \
    --print_freq 100 \
    --port 18372 > $asr_exp/train.log 2>&1 &
    
  echo "[Stage 1] ASR Training Finished."
fi

out_name='averaged.mdl'
if [ $stage -le 2 ] && [ $end_stage -ge 2 ]; then
  last_epoch=79  # Need to be modified according to the convergence
  
  average_checkpoints.py \
    --exp_dir $asr_exp \
    --out_name $out_name \
    --last_epoch $last_epoch \
    --num 10
  
  echo "[Stage 2] Average checkpoints Finished."

fi

if [ $stage -le 3 ] && [ $end_stage -ge 3 ]; then
  asr_exp=exp/1kh_conformer_rel_maxlen20_e10d5_accum2_specaug_tmax10_multistep2k_40k_160k_ln/
  exp=$asr_exp
  lm_exp=exp/libri_tfunilm16x512_4card_cosineanneal_ep20_maxlen120/


  test_model=$exp/$out_name
  #rnnlm_model=$lm_exp/$out_name
  rnnlm_model=$lm_exp/averaged.mdl
  decode_type='ctc_att'
  attbeam=20  # set in conf/decode.yaml, att beam
  ctcbeam=30  # set in conf/decode.yaml, ctc beam
  lp=0        #set in conf/decode.yaml, length penalty
  ctcwt=0.4
  lmwt=0.6
  nj=16
  batch_size=1
  test_set="test_clean test_other dev_clean dev_other"
  
  for tset in $test_set; do
    echo "Decoding $tset..."
    desdir=$exp/${decode_type}_decode_ctc${ctcwt}_attbm_${attbeam}_ctcbm_${ctcbeam}_lp${lp}_newlmwt${lmwt}_lmeos/$tset/

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


