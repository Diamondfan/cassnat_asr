. cmd.sh
. path.sh

stage=1
end_stage=1

asr_exp=exp/myst_sp_hubert_art_lr5e-5_1e-3_samples_bs8_accum4/

if [ $stage -le 1 ] && [ $end_stage -ge 1 ]; then

  if [ ! -d $asr_exp ]; then
    mkdir -p $asr_exp
  fi

  CUDA_VISIBLE_DEVICES="0,1" train_asr.py \
    --task "hubert_art" \
    --exp_dir $asr_exp \
    --train_config conf/hubert_art_train.yaml \
    --data_config conf/data_hubert.yaml \
    --optim_type "noam" \
    --epochs 60 \
    --start_saving_epoch 1 \
    --end_patience 10 \
    --seed 1234 \
    --port 11234 \
    --print_freq 100 > $asr_exp/train.log 2>&1 &  
    
  echo "[Stage 1] ASR Training Finished."
fi

out_name='averaged.mdl'
if [ $stage -le 2 ] && [ $end_stage -ge 2 ]; then
  last_epoch=22  # need to be modified
  
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
  bpemodel=data/dict/bpemodel_unigram_1024
  decode_type='ctc_att'
  attbeam=20  # set in conf/decode.yaml, att beam
  ctcbeam=20  # set in conf/decode.yaml, ctc beam
  lp=0        #set in conf/decode.yaml, length penalty
  ctcwt=0.4
  lmwt=0
  nj=1
  batch_size=1
  test_set="dev_clean dev_other test_clean test_other"
  
  for tset in $test_set; do
    echo "Decoding $tset..."
    desdir=$exp/${decode_type}_decode_ctc${ctcwt}_attbm_${attbeam}_ctcbm_${ctcbeam}_lp${lp}_lmwt${lmwt}/$tset/

    if [ ! -d $desdir ]; then
      mkdir -p $desdir
    fi
    
    split_scps=
    for n in $(seq $nj); do
      split_scps="$split_scps $desdir/wav_s.$n.scp"
    done
    utils/split_scp.pl data/$tset/wav_s.scp $split_scps || exit 1;
    
    $cmd JOB=1:$nj $desdir/log/decode.JOB.log \
      CUDA_VISIBLE_DEVICES=0 decode_asr.py \
        --task "hubert_art" \
        --test_config conf/hubert_art_decode.yaml \
        --lm_config conf/lm.yaml \
        --data_path $desdir/wav_s.JOB.scp \
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
  echo "[Stage 3] Hubert ART Decoding Finished."
fi

