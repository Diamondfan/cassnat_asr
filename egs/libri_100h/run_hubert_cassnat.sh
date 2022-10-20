. cmd.sh
. path.sh

stage=3
end_stage=3

asr_exp=exp/hubert_cassnat_lr5e-5_1e-3_bs4_accum4/

if [ $stage -le 1 ] && [ $end_stage -ge 1 ]; then

  if [ ! -d $asr_exp ]; then
    mkdir -p $asr_exp
  fi

  CUDA_VISIBLE_DEVICES="0,1" train_asr.py \
    --task "hubert_cassnat" \
    --exp_dir $asr_exp \
    --train_config conf/hubert_cassnat_train.yaml \
    --data_config conf/data_hubert.yaml \
    --optim_type "noam" \
    --epochs 60 \
    --start_saving_epoch 1 \
    --end_patience 10 \
    --seed 1234 \
    --port 12345 \
    --print_freq 100 >> $asr_exp/train.log 2>&1 &  
    
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
  rank_model="at_baseline"
  #rnnlm_model=exp/libri_tfunilm16x512_4card_cosineanneal_ep20_maxlen120/averaged.mdl
  rnnlm_model=exp/hubert_art_lr5e-5_1e-3_bs4_accum4/averaged.mdl
  rank_yaml=conf/hubert_rank_model.yaml
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
  nj=1
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
      split_scps="$split_scps $desdir/wav_s.$n.scp"
    done
    utils/split_scp.pl data/$tset/wav_s.scp $split_scps || exit 1;

    $cmd JOB=1:$nj $desdir/log/decode.JOB.log \
      CUDA_VISIBLE_DEVICES=JOB decode_asr.py \
        --task "hubert_cassnat" \
        --test_config conf/hubert_cassnat_decode.yaml \
        --lm_config $rank_yaml \
        --rank_model $rank_model \
        --data_path $desdir/wav_s.JOB.scp \
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

