
# 2023 Ruchao Fan, SPAPL, UCLA

. cmd.sh
. path.sh

stage=3
end_stage=3

# unienncoder cassnat settings
#train_config=conf/unienc_cassnat_train.yaml
#data_config=conf/data_raw.yaml

# cassnat with hubert encoder settings
train_config=conf/hubert_unienc_cassnat_train.yaml
data_config=conf/data_hubert.yaml

#asr_exp=exp/unienc_cassnat_lr1e-3_specaugt5_004_multictc/ #_maskt03f05/ 
#asr_exp=exp/hubert_unienc_cassnat_lr5e-5_1e-3_samples_max60s_accum4_warmup12k_maskt04f05_singlectc/
asr_exp=exp/hubert_unienc_cassnat_lr5e-5_1e-3_samples_max60s_accum4_warmup12k_maskt04f05_multictc1_d768f3072/

if [ $stage -le 1 ] && [ $end_stage -ge 1 ]; then

  [ ! -d $asr_exp ] && mkdir -p $asr_exp;

  CUDA_VISIBLE_DEVICES="0,1" train_asr.py \
    --task "unienc_cassnat" \
    --exp_dir $asr_exp \
    --train_config $train_config \
    --data_config $data_config \
    --optim_type "noam" \
    --epochs 60 \
    --start_saving_epoch 5 \
    --end_patience 10 \
    --seed 1234 \
    --port 21251 \
    --print_freq 100 >> $asr_exp/train.log 2>&1 &  
    
  echo "[Stage 1] ASR Training Finished."
fi

out_name='averaged.mdl'
if [ $stage -le 2 ] && [ $end_stage -ge 2 ]; then
  last_epoch=20  # need to be modified
  
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
  #rnnlm_model=exp/100h_sptokenizer_cfmer_interctc05_layer6_noam_warmup15k_lrpk1e-3_epoch60_2gpus/averaged.mdl
  #rank_yaml=conf/rank_model.yaml
  rnnlm_model=exp/hubert_art_lr5e-5_1e-3_samples_max80s_accum4_warmup10k_maskt06f05/averaged.mdl
  rank_yaml=conf/hubert_rank_model.yaml
  test_model=$asr_exp/$out_name
  decode_type='esa_att'
  attbm=1
  ctcbm=1 
  ctclm=0
  ctclp=0
  lmwt=0
  s_num=50
  s_num2=1
  threshold=0.9
  s_dist=0
  lp=0
  nj=1
  iters=1
  batch_size=1
  test_set="dev_clean dev_other" #test_clean test_other" #

  # decode cassnat model
  #decode_config=conf/unienc_cassnat_decode.yaml
  #data_prefix=feats

  # decode cassnat model with hubert encoder
  decode_config=conf/hubert_unienc_cassnat_decode.yaml
  data_prefix=wav_s

  for tset in $test_set; do
    echo "Decoding $tset..."
    desdir=$exp/${decode_type}_iters${iters}_decode_attbm_${attbm}_sampdist_${s_dist}_samplenum_${s_num}_samplenum2_${s_num2}_lm${lmwt}_threshold${threshold}_rank${rank_model}/$tset/

    if [ ! -d $desdir ]; then
      mkdir -p $desdir
    fi
    
    split_scps=
    for n in $(seq $nj); do
      split_scps="$split_scps $desdir/${data_prefix}.$n.scp"
    done
    utils/split_scp.pl data/$tset/${data_prefix}.scp $split_scps || exit 1;
    
    $cmd JOB=1:$nj $desdir/log/decode.JOB.log \
      CUDA_VISIBLE_DEVICES="2" decode_asr.py \
        --task "unienc_cassnat" \
        --test_config $decode_config \
        --lm_config $rank_yaml \
        --rank_model $rank_model \
        --data_path $desdir/${data_prefix}.JOB.scp \
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

