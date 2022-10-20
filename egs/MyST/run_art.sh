#!/usr/bin/env bash

# 2020 (Ruchao Fan)
# 2022 (Ruchao Fan)

stage=3
end_stage=3
featdir=data/fbank

unit=wp         #word piece
nbpe=500
bpemode=unigram #bpe or unigram

. ./cmd.sh
. ./path.sh
. parse_options.sh

asr_exp=exp/myst_sp_sptokenizer_cfmer_interctc05_layer6_noam_warmup25k_lrpk1e-3_epoch60_2gpus/

if [ $stage -le 1 ] && [ $end_stage -ge 1 ]; then

  [ ! -d $asr_exp ] && mkdir -p $asr_exp

  CUDA_VISIBLE_DEVICES="2,3" train_asr.py \
    --task "art" \
    --exp_dir $asr_exp \
    --train_config conf/transformer.yaml \
    --data_config conf/data_raw.yaml \
    --optim_type "noam" \
    --epochs 60 \
    --start_saving_epoch 30 \
    --end_patience 10 \
    --seed 1234 \
    --print_freq 100 \
    --port 13456 > $asr_exp/train.log 2>&1 &
    
  echo "[Stage 1] ASR Training Finished."
fi

out_name='averaged.mdl'
if [ $stage -le 2 ] && [ $end_stage -ge 2 ]; then
  last_epoch=35  # Need to be modified according to the convergence
  
  average_checkpoints.py \
    --exp_dir $asr_exp \
    --out_name $out_name \
    --last_epoch $last_epoch \
    --num 5
  
  echo "[Stage 2] Average checkpoints Finished."

fi

if [ $stage -le 3 ] && [ $end_stage -ge 3 ]; then
  exp=$asr_exp
  lm_exp=
  test_model=$exp/$out_name
  rnnlm_model=$lm_exp/averaged.mdl
  bpemodel=data/dict/bpemodel_${bpemode}_${nbpe}
  decode_type='ctc_att'
  attbeam=20  # set in conf/decode.yaml, att beam
  ctcbeam=20  # set in conf/decode.yaml, ctc beam
  lp=0        #set in conf/decode.yaml, length penalty
  ctcwt=0.4
  lmwt=0
  nj=1
  batch_size=1
  test_set="development test"
  
  for tset in $test_set; do
    echo "Decoding $tset..."
    desdir=$exp/${decode_type}_decode_ctc${ctcwt}_attbm_${attbeam}_ctcbm_${ctcbeam}_lp${lp}_lmwt${lmwt}/$tset/

    if [ ! -d $desdir ]; then
      mkdir -p $desdir
    fi
    
    utils/split_data.sh data/$tset/ $nj || exit 1;
    
    $cmd JOB=1:$nj $desdir/log/decode.JOB.log \
      CUDA_VISIBLE_DEVICES=2 decode_asr.py \
        --task "art" \
        --test_config conf/decode.yaml \
        --lm_config conf/lm.yaml \
        --data_path data/$tset/split$nj/JOB/feats.scp \
        --utt2num_frames data/$tset/split$nj/JOB/utt2num_frames \
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
  echo "[Stage 3] Decoding Finished."
fi


