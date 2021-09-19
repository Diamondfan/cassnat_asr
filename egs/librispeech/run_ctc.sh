
stage=1
end_stage=1
featdir=data/fbank

. ./cmd.sh
. ./path.sh
. parse_options.sh

asr_exp=exp_ctc/1kh_wp5k_tsfm_noam10k_pk2e-3_torch1.2/ #_shareff/

if [ $stage -le 1 ] && [ $end_stage -ge 1 ]; then

  if [ ! -d $asr_exp ]; then
    mkdir -p $asr_exp
  fi  

  CUDA_VISIBLE_DEVICES="4,5,6,7" ctc_train.py \
    --exp_dir $asr_exp \
    --train_config conf/ctc.yaml \
    --data_config conf/data_ctc.yaml \
    --epochs 100 \
    --save_epoch 40 \
    --learning_rate 0.002 \
    --min_lr 0.00001 \
    --end_patience 10 \
    --opt_type "noam" \
    --weight_decay 0 \
    --ctc_alpha 1 \
    --interctc_alpha 0 \
    --use_cmvn \
    --seed 1234 \
    --print_freq 50 #> $asr_exp/train.log 2>&1 &
    
  echo "[Stage 6] ASR Training Finished."
fi
