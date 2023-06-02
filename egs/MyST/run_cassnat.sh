#!/usr/bin/env bash

# 2020 Ruchao Fan PAII Inc.
# 2022 Ruchao Fan SPAPL UCLA
# This script is to run our proposed CASS-NAT, The name is CASS-NAT
# (CTC alignement-based Signgle Step Non-autoregressive Transformer).


. cmd.sh
. path.sh

stage=1
end_stage=1
lm_model=""

# cassnat settings
train_config=conf/cassnat_train.yaml
data_config=conf/data_raw.yaml
start_saving_epoch=10

# cassnat with hubert encoder settings
#train_config=conf/hubert_cassnat_train.yaml
#data_config=conf/data_hubert.yaml
#start_saving_epoch=5

asr_exp=exp/mystsp_sptokenizer_cassnat_initart_utterance_noam_warmup25k/
#asr_exp=exp/mystsp_sptokenizer_hubert_cassnat_samples_noam_max800k/

if [ $stage -le 1 ] && [ $end_stage -ge 1 ]; then

  [ ! -d $asr_exp ] && mkdir -p $asr_exp;

  CUDA_VISIBLE_DEVICES="0,1" train_asr.py \
    --task "cassnat" \
    --exp_dir $asr_exp \
    --train_config $train_config \
    --data_config $data_config \
    --optim_type "noam" \
    --epochs 60 \
    --start_saving_epoch $start_saving_epoch \
    --end_patience 10 \
    --seed 1234 \
    --print_freq 100 \
    --port 25142 > $asr_exp/train.log 2>&1 &
    
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

  bpemodel=data/dict/bpemodel_unigram_500
  rank_model="at_baseline" #"lm", "at_baseline"
  #rnnlm_model=$lm_model
  rnnlm_model=exp/mystsp_sptokenizer_art_conformer_utterance_noam_warmup25k/averaged.mdl
  rank_yaml=conf/rank_model.yaml
  #rnnlm_model=exp/mystsp_sptokenizer_hubert_art_samples_noam_max800k/averaged.mdl
  #rank_yaml=conf/hubert_rank_model.yaml
  test_model=$asr_exp/$out_name
  decode_type='esa_att'
  attbm=1
  ctcbm=1 
  ctclm=0
  ctclp=0
  lmwt=0
  s_num=16
  threshold=0.9
  s_dist=0
  lp=0
  nj=1
  batch_size=1
  test_set="test" #development" #test"

  # decode cassnat model
  decode_config=conf/cassnat_decode.yaml
  data_prefix=feats

  # decode cassnat model with hubert encoder
  #decode_config=conf/hubert_cassnat_decode.yaml
  #data_prefix=wav_s

  for tset in $test_set; do
    echo "Decoding $tset..."
    desdir=$exp/${decode_type}_decode_attbm_${attbm}_sampdist_${s_dist}_samplenum_${s_num}_lm${lmwt}_threshold${threshold}_rank${rank_model}/$tset/

    [ ! -d $desdir ] && mkdir -p $desdir;
    
    split_scps=
    for n in $(seq $nj); do
      split_scps="$split_scps $desdir/${data_prefix}.$n.scp"
    done
    utils/split_scp.pl data/$tset/${data_prefix}.scp $split_scps || exit 1;
    
    $cmd JOB=1:$nj $desdir/log/decode.JOB.log \
      CUDA_VISIBLE_DEVICES="3" decode_asr.py \
        --task "cassnat" \
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

if [ $stage -le 4 ] && [ $end_stage -ge 4 ]; then
    test_set="development test"
    for tset in $test_set; do
        #UM, ER, UH, AH, and HMM
        #desdir=exp/mystsp_sptokenizer_art_conformer_utterance_noam_warmup25k/ctc_att_decode_ctc0.4_attbm_20_ctcbm_20_lp0_lmwt0/$tset/
        desdir=exp/mystsp_sptokenizer_cassnat_initrand_utterance_noam_warmup25k//esa_att_decode_attbm_1_sampdist_0_samplenum_16_lm0_threshold0.9_rankat_baseline/$tset/
        for file in $desdir/ref.wrd.trn $desdir/hyp.wrd.trn; do
            new_file=`echo $file | sed "s:wrd:clnwrd:g"`
            python local/remove_fill_pauses.py $file $new_file
        done 
        sclite -r $desdir/ref.clnwrd.trn -h $desdir/hyp.clnwrd.trn -i rm -o all stdout > $desdir/result.clnwrd.txt
    done
    echo "[Stage 4] Remove Filter Pauses and Compute WER again!"
fi

