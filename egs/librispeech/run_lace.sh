
. cmd.sh
. path.sh

stage=1
end_stage=1

if [ $stage -le 1 ] && [ $end_stage -ge 1 ]; then
  exp=exp/1kh_small_lace_alm0_6bi_ctc1_att1/

  if [ ! -d $exp ]; then
    mkdir -p $exp
  fi

  CUDA_VISIBLE_DEVICES="0" lace_train.py \
    --exp_dir $exp \
    --train_config conf/transformer.yaml \
    --data_config conf/data.yaml \
    --batch_size 32 \
    --epochs 30 \
    --save_epoch 8 \
    --anneal_lr_epoch 9 \
    --anneal_lr_ratio 0.5 \
    --n_anneal 4 \
    --learning_rate 0.0002 \
    --opt_type "normal" \
    --weight_decay 0 \
    --label_smooth 0.1 \
    --ctc_alpha 1 \
    --alm_alpha 0 \
    --print_freq 200 > $exp/train.log 2>&1 &
    
  echo "[Stage 1] ASR Training Finished."
fi


if [ $stage -le 2 ] && [ $end_stage -ge 2 ]; then
  #exp=exp/1kh_small_lace_quick/
  exp=exp/1kh_small_lace_alm0_3uni_3bi/

  bpemodel=data/dict/bpemodel_bpe_5000
  test_model=$exp/best_model.mdl
  decode_type='att_only'
  beam1=0 # check beam1 and beam2 in conf/decode.yaml, att beam
  beam2=0 # ctc beam
  ctcwt=0
  lmwt=0
  lp=0.2
  nj=4
  
  for tset in test_clean; do #$test_set; do
    echo "Decoding $tset..."
    desdir=$exp/${decode_type}_decode_ctc${ctcwt}_bm1_${beam1}_bm2_${beam2}_lmwt${lmwt}_lp${lp}/$tset/
    if [ ! -d $desdir ]; then
      mkdir -p $desdir
    fi
    
    split_scps=
    for n in $(seq $nj); do
      split_scps="$split_scps $desdir/feats.$n.scp"
    done
    utils/split_scp.pl data/$tset/feats.scp $split_scps || exit 1;
    
    $cmd JOB=1:$nj $desdir/log/decode.JOB.log \
      CUDA_VISIBLE_DEVICES="0" lace_decode.py \
        --test_config conf/decode.yaml \
        --data_path $desdir/feats.JOB.scp \
        --resume_model $test_model \
        --result_file $desdir/token_results.JOB.txt \
        --batch_size 8 \
        --decode_type $decode_type \
        --ctc_weight $ctcwt \
        --rnnlm None \
        --lm_weight $lmwt \
        --max_decode_ratio 0 \
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
