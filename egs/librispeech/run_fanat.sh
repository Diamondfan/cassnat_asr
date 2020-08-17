
. cmd.sh
. path.sh

stage=3
end_stage=3

if [ $stage -le 1 ] && [ $end_stage -ge 1 ]; then
  exp=exp/fanat_large_specaug_multistep_trig_src/

  if [ ! -d $exp ]; then
    mkdir -p $exp
  fi

  CUDA_VISIBLE_DEVICES="0,1,2,3" fanat_train.py \
    --exp_dir $exp \
    --train_config conf/fanat_train.yaml \
    --data_config conf/data.yaml \
    --batch_size 16 \
    --epochs 100 \
    --save_epoch 40 \
    --anneal_lr_ratio 0.5 \
    --patience 1 \
    --end_patience 10 \
    --learning_rate 0.001 \
    --min_lr 0.00001 \
    --opt_type "multistep" \
    --weight_decay 0 \
    --label_smooth 0.1 \
    --ctc_alpha 1 \
    --embed_alpha 0 \
    --use_cmvn \
    --print_freq 100 > $exp/train.log 2>&1 &
    
    #--init_encoder \
    #--resume_model exp/baseline/averaged.mdl \
    #--word_embed exp/averaged.mdl \
    #--knowlg_dist \
  echo "[Stage 1] ASR Training Finished."
fi

out_name='averaged.mdl'
if [ $stage -le 2 ] && [ $end_stage -ge 2 ]; then
  exp=exp/fanat_large_specaug_multistep_trig_src
  last_epoch=89
  
  average_checkpoints.py \
    --exp_dir $exp \
    --out_name $out_name \
    --last_epoch $last_epoch \
    --num 12
  
  echo "[Stage 2] Average checkpoints Finished."

fi

if [ $stage -le 3 ] && [ $end_stage -ge 3 ]; then
  exp=exp/fanat_large_specaug_multistep_trig_src

  bpemodel=data/dict/bpemodel_unigram_5000
  rnnlm_model=exp/libri_tflm_unigram_4card_cosineanneal_ep10/$out_name
  global_cmvn=data/fbank/cmvn.ark
  test_model=$exp/$out_name
  decode_type='att_only'
  beam1=1 #5 # check beam1 and beam2 in conf/decode.yaml, att beam
  beam2=0 #10 # ctc beam
  ctcwt=0
  lmwt=0 #0.5
  ctclm=0 #0.7
  ctclp=0 #2
  lp=0
  nj=2
  test_set="dev_clean test_clean dev_other test_other"

  for tset in $test_set; do
    echo "Decoding $tset..."
    #desdir=$exp/${decode_type}_decode_average_ctc${ctcwt}_bm1_${beam1}_bm2_${beam2}_lmwt${lmwt}_ctclm${ctclm}_ctclp${ctclp}_lp${lp}/$tset/
    desdir=$exp/${decode_type}_decode_average_bm1_${beam1}_sampdist_1_samplenum_20/$tset/

    if [ ! -d $desdir ]; then
      mkdir -p $desdir
    fi
    
    split_scps=
    for n in $(seq $nj); do
      split_scps="$split_scps $desdir/feats.$n.scp"
    done
    utils/split_scp.pl data/$tset/feats.scp $split_scps || exit 1;
    
    $cmd JOB=1:$nj $desdir/log/decode.JOB.log \
      CUDA_VISIBLE_DEVICES="1" fanat_decode.py \
        --test_config conf/fanat_decode.yaml \
        --lm_config conf/lm.yaml \
        --data_path $desdir/feats.JOB.scp \
        --text_label data/$tset/token.scp \
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
fi
