
. cmd.sh
. path.sh

stage=1
end_stage=1

encoder_initial_model=exp/ar_convenc_best_interctc05_ctc05/averaged.mdl
asr_exp=exp/conv_fanat_convdec_maxlen4_interctc05_interce01_ce09_aftermapping/

if [ $stage -le 1 ] && [ $end_stage -ge 1 ]; then

  if [ ! -d $asr_exp ]; then
    mkdir -p $asr_exp
  fi

  CUDA_VISIBLE_DEVICES="0,1,2,3" fanat_train.py \
    --exp_dir $asr_exp \
    --train_config conf/fanat_train.yaml \
    --data_config conf/data.yaml \
    --batch_size 32 \
    --epochs 100 \
    --save_epoch 30 \
    --end_patience 10 \
    --learning_rate 0.001 \
    --min_lr 0.00001 \
    --opt_type "multistep" \
    --weight_decay 0 \
    --label_smooth 0.1 \
    --ctc_alpha 0.5 \
    --interctc_alpha 0.5 \
    --att_alpha 0.9 \
    --interce_alpha 0.1 \
    --interce_location 'after_mapping' \
    --use_cmvn \
    --init_encoder \
    --resume_model $encoder_initial_model \
    --seed 1234 \
    --print_freq 50 > $asr_exp/train.log 2>&1 &
    
  echo "[Stage 1] ASR Training Finished."
fi

out_name='averaged.mdl'
if [ $stage -le 2 ] && [ $end_stage -ge 2 ]; then
  last_epoch=69
  
  average_checkpoints.py \
    --exp_dir $asr_exp \
    --out_name $out_name \
    --last_epoch $last_epoch \
    --num 12
  
  echo "[Stage 2] Average checkpoints Finished."

fi

if [ $stage -le 3 ] && [ $end_stage -ge 3 ]; then
  exp=$asr_exp

  rnnlm_model=exp/ar_convenc_best_interctc05_ctc05/averaged.mdl
  global_cmvn=data/fbank/cmvn.ark
  test_model=$exp/$out_name
  decode_type='att_only'
  beam1=1
  beam2=1 
  ctcwt=0
  lmwt=0
  ctclm=0
  ctclp=0
  s_num=50
  s_dist=0
  lp=0
  nj=4
  batch_size=4
  test_set="dev test"

  for tset in $test_set; do
    echo "Decoding $tset..."
    desdir=$exp/${decode_type}_decode_average_bm1_${beam1}_sampdist_${s_dist}_samplenum_${s_num}_newlm${lmwt}/$tset/

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
fi
