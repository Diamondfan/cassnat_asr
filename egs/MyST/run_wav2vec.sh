#!/usr/bin/env bash

# 2020 (Ruchao Fan)
# 2022 (Ruchao Fan)

stage=1
end_stage=1
featdir=data/fbank

. ./cmd.sh
. ./path.sh
. parse_options.sh

asr_exp=exp_ssl/test2 #conformer_divloss02_3gpus_accum12_400eps/

if [ $stage -le 1 ] && [ $end_stage -ge 1 ]; then

  [ ! -d $asr_exp ] && mkdir -p $asr_exp

  CUDA_VISIBLE_DEVICES="0,1,2" train_ssl.py \
    --task "wav2vec" \
    --exp_dir $asr_exp \
    --train_config conf/ssl_wav2vec.yaml \
    --data_config conf/data_ssl.yaml \
    --optim_type "noam" \
    --epochs 400 \
    --start_saving_epoch 350 \
    --end_patience 400 \
    --seed 1234 \
    --print_freq 50 \
    --port 18372 #> $asr_exp/train.log 2>&1 &
    
  echo "[Stage 1] ASR Training Finished."
fi

out_name='averaged.mdl'
if [ $stage -le 2 ] && [ $end_stage -ge 2 ]; then
  last_epoch=79  # Need to be modified according to the convergence
  
  average_checkpoints.py \
    --exp_dir $asr_exp \
    --out_name $out_name \
    --last_epoch $last_epoch \
    --num 10
  
  echo "[Stage 2] Average checkpoints Finished."

fi


