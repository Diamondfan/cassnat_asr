
. cmd.sh
. path.sh

stage=3
end_stage=3

if [ $stage -le 1 ] && [ $end_stage -ge 1 ]; then
  #exp=exp/fanat_multistep_notrig_nosrc_nouni_ctc1/
  exp=exp/fanat_large_specaug_multistep_trig_src_initenc

  if [ ! -d $exp ]; then
    mkdir -p $exp
  fi

  CUDA_VISIBLE_DEVICES="4,5,6,7" fanat_train.py \
    --exp_dir $exp \
    --train_config conf/fanat_train.yaml \
    --data_config conf/data.yaml \
    --batch_size 32 \
    --epochs 100 \
    --save_epoch 30 \
    --anneal_lr_ratio 0.5 \
    --patience 1 \
    --end_patience 10 \
    --learning_rate 0.001 \
    --min_lr 0.00001 \
    --opt_type "multistep" \
    --weight_decay 0 \
    --label_smooth 0.1 \
    --ctc_alpha 1 \
    --init_encoder \
    --resume_model exp/1kh_d512_multistep_ctc1_accum1_bth32_specaug/averaged.mdl \
    --use_cmvn \
    --print_freq 50 > $exp/train.log 2>&1 &
    
    #--embed_loss_type 'l2' \
    #--init_encoder \
    #--knowlg_dist \
  echo "[Stage 1] ASR Training Finished."
fi

out_name='averaged.mdl'
if [ $stage -le 2 ] && [ $end_stage -ge 2 ]; then
  exp=exp/fanat_large_specaug_multistep_trig_src_initenc
  last_epoch=93
  
  average_checkpoints.py \
    --exp_dir $exp \
    --out_name $out_name \
    --last_epoch $last_epoch \
    --num 12
  
  echo "[Stage 2] Average checkpoints Finished."

fi

if [ $stage -le 3 ] && [ $end_stage -ge 3 ]; then
  exp=exp/fanat_large_specaug_multistep_trig_src_initenc

  rnnlm_model=exp/1kh_d512_multistep_ctc1_accum1_bth32_specaug/averaged.mdl
  global_cmvn=data/fbank/cmvn.ark
  test_model=$exp/$out_name
  decode_type='oracle_att' #_only'
  beam1=1 # check beam1 and beam2 in conf/decode.yaml, att beam
  beam2=0 #10 # ctc beam
  ctcwt=0
  lmwt=0
  ctclm=0 #0.7
  ctclp=0 #2
  lp=0
  s_dist=0
  s_num=0
  nj=4
  test_set="dev test"

  for tset in $test_set; do
    echo "Decoding $tset..."
    desdir=$exp/${decode_type}_decode_average_ctc${ctcwt}_bm1_${beam1}_bm2_${beam2}_lmwt${lmwt}_ctclm${ctclm}_ctclp${ctclp}_lp${lp}/$tset/
    #desdir=$exp/${decode_type}_decode_average_bm1_${beam1}_sampdist_${s_dist}_samplenum_${s_num}_newlm${lmwt}/$tset/
    if [ ! -d $desdir ]; then
      mkdir -p $desdir
    fi
    
    split_scps=
    for n in $(seq $nj); do
      split_scps="$split_scps $desdir/feats.$n.scp"
    done
    utils/split_scp.pl data/$tset/feats.scp $split_scps || exit 1;
    
    $cmd JOB=1:$nj $desdir/log/decode.JOB.log \
      CUDA_VISIBLE_DEVICES=JOB fanat_decode.py \
        --test_config conf/fanat_decode.yaml \
        --lm_config conf/decode.yaml \
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
        --word_embed exp/1kh_d512_multistep_ctc1_accum1_bth32_specaug/averaged.mdl \
        --global_cmvn $global_cmvn \
        --print_freq 20 

    cat $desdir/token_results.*.txt | sort -k1,1 > $desdir/token_results.txt
    text2trn.py $desdir/token_results.txt $desdir/hyp.token.trn
    text2trn.py data/$tset/token.scp $desdir/ref.token.trn
 
    sclite -r $desdir/ref.token.trn -h $desdir/hyp.token.trn -i wsj -o all stdout > $desdir/result.wrd.txt
  done
fi
