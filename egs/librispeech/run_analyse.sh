#!/usr/bin/env bash

# 2021 Ruchao Fan
# This script is to make an anaysis for CASS-NAT model.
# Attention plot
# Token-level embedding analysis

. cmd.sh
. path.sh

stage=1
end_stage=1

#asr_exp=exp/conv_fanat_e10m2d4_max_specaug_multistep_initenc_convdec_maxlen8_kernel3_ctxtrig1/
asr_exp=exp/conv_fanat_best_interctc05_ctc05_interce01_ce09/

if [ $stage -le 1 ] && [ $end_stage -ge 1 ]; then
  exp=$asr_exp
  out_name=averaged.mdl

  bpemodel=data/dict/bpemodel_unigram_5000
  #rnnlm_model=$lm_model
  rnnlm_model=exp_cassnat/tf_unilm/averaged.mdl
  global_cmvn=data/fbank/cmvn.ark
  test_model=$asr_exp/$out_name
  beam1=1
  beam2=1 
  lmwt=0
  ctclm=0
  ctclp=0
  s_num=50
  s_dist=0
  lp=0
  batch_size=1
  test_set="train_clean_100" #test_other dev_clean dev_other"

  for tset in $test_set; do
    echo "Decoding $tset..."
    desdir=$exp/test #${decode_type}_decode_average_bm1_${beam1}_sampdist_${s_dist}_samplenum_${s_num}_newlm${lmwt}/$tset/

    if [ ! -d $desdir ]; then
      mkdir -p $desdir
    fi
    
    CUDA_VISIBLE_DEVICES='1' fanat_analyze.py \
      --test_config conf/fanat_decode.yaml \
      --lm_config conf/lm.yaml \
      --rank_model "lm" \
      --data_path data/$tset/feats.scp \
      --text_label data/$tset/token.scp \
      --resume_model $test_model \
      --result_file $desdir/token_results.txt \
      --batch_size $batch_size \
      --rnnlm $rnnlm_model \
      --lm_weight $lmwt \
      --max_decode_ratio 0 \
      --use_cmvn \
      --save_embedding \
      --global_cmvn $global_cmvn \
      --print_freq 20

    #cat $desdir/token_results.*.txt | sort -k1,1 > $desdir/token_results.txt
    #text2trn.py $desdir/token_results.txt $desdir/hyp.token.trn
    #text2trn.py data/$tset/token.scp $desdir/ref.token.trn
 
    #spm_decode --model=${bpemodel}.model --input_format=piece < $desdir/hyp.token.trn | sed -e "s/▁/ /g" |\
    #        sed -e "s/(/ (/g" > $desdir/hyp.wrd.trn
    #spm_decode --model=${bpemodel}.model --input_format=piece < $desdir/ref.token.trn | sed -e "s/▁/ /g" |\
    #        sed -e "s/(/ (/g" > $desdir/ref.wrd.trn
    #sclite -r $desdir/ref.wrd.trn -h $desdir/hyp.wrd.trn -i rm -o all stdout > $desdir/result.wrd.txt
  done
fi


