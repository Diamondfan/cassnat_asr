# 2022 Ruchao Fan

stage=1
end_stage=1

. ./cmd.sh
. ./path.sh
. parse_options.sh

asr_exp=exp/100h_test/
unit=wp
bpemodel=data/dict/bpemodel_unigram_1024

if [ $stage -le 1 ] && [ $end_stage -ge 1 ]; then

  if [ ! -d $asr_exp ]; then
    mkdir -p $asr_exp
  fi  

  CUDA_VISIBLE_DEVICES="2" train_asr.py \
    --task "ctc" \
    --exp_dir $asr_exp \
    --train_config conf/ctc.yaml \
    --data_config conf/data_wp.yaml \
    --optim_type "noam" \
    --epochs 80 \
    --start_saving_epoch 40 \
    --end_patience 10 \
    --seed 1234 \
    --print_freq 100 \
    --port 12387 #> $asr_exp/train.log 2>&1 &
    
  echo "[Stage 6] ASR Training Finished."
fi

out_name='averaged.mdl'
if [ $stage -le 2 ] && [ $end_stage -ge 2 ]; then
  last_epoch=99  # Need to be modified according to the convergence
  
  average_checkpoints.py \
    --exp_dir $asr_exp \
    --out_name $out_name \
    --last_epoch $last_epoch \
    --num 12
  
  echo "[Stage 2] Average checkpoints Finished."

fi

if [ $stage -le 3 ] && [ $end_stage -ge 3 ]; then
  exp=$asr_exp

  test_model=$exp/$out_name
  rnnlm_model=exp_cassnat/tf_unilm/averaged.mdl
  decode_type='greedy'
  beam=1 #20     # set in conf/ctc_decode.yaml
  pruning=1 #33  # set in conf/ctc_decode.yaml
  lp=0 #2        # set in conf/ctc_decode.yaml, length penalty
  lmwt=0
  nj=8
  batch_size=1
  test_set="test_clean test_other dev_clean dev_other"

  for tset in $test_set; do
    echo "Decoding $tset..."
    desdir=$exp/${decode_type}_decode_bm_${beam}_pruning_${pruning}_lmwt${lmwt}_lp${lp}/$tset/

    if [ ! -d $desdir ]; then
      mkdir -p $desdir
    fi
    
    split_scps=
    for n in $(seq $nj); do
      split_scps="$split_scps $desdir/feats.$n.scp"
    done
    utils/split_scp.pl data/$tset/feats.scp $split_scps || exit 1;
    
    $cmd JOB=1:$nj $desdir/log/decode.JOB.log \
      CUDA_VISIBLE_DEVICES=JOB ctc_decode.py \
        --task "ctc" \
        --test_config conf/ctc_decode.yaml \
        --lm_config conf/lm.yaml \
        --data_path $desdir/feats.JOB.scp \
        --resume_model $test_model \
        --result_file $desdir/token_results.JOB.txt \
        --batch_size $batch_size \
        --rnnlm $rnnlm_model \
        --lm_weight $lmwt \
        --print_freq 20 
    
    cat $desdir/token_results.*.txt | sort -k1,1 > $desdir/token_results.txt
    text2trn.py $desdir/token_results.txt $desdir/hyp.token.trn
 
    if [ $unit == wp ]; then
      text2trn.py data/$tset/token.scp $desdir/ref.token.trn
      spm_decode --model=${bpemodel}.model --input_format=piece < $desdir/hyp.token.trn | sed -e "s/▁/ /g" |\
            sed -e "s/(/ (/g" > $desdir/hyp.wrd.trn
      spm_decode --model=${bpemodel}.model --input_format=piece < $desdir/ref.token.trn | sed -e "s/▁/ /g" |\
            sed -e "s/(/ (/g" > $desdir/ref.wrd.trn

    elif [ $unit == char ]; then
      text2trn.py data/$tset/token_char.scp $desdir/ref.token.trn
      cat $desdir/hyp.token.trn | sed -e "s: ::g" | sed -e "s/|/ /g" | sed -e "s/(/ (/g" > $desdir/hyp.wrd.trn
      cat $desdir/ref.token.trn | sed -e "s: ::g" | sed -e "s/|/ /g" | sed -e "s/(/ (/g" > $desdir/ref.wrd.trn
    else
      echo "Not ImplementedError"; exit 1
    fi

    sclite -r $desdir/ref.wrd.trn -h $desdir/hyp.wrd.trn -i rm -o all stdout > $desdir/result.wrd.txt

  done
  echo "[Stage 7] Decoding Finished."
fi


