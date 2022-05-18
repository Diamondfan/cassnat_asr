
stage=1
end_stage=1

. ./path.sh

lm_data=/data/Databases/LibriSpeech/libri_lm

if [ $stage -le 1 ] && [ $end_stage -ge 1 ]; then
  echo "stage 1: LM Preparation"
  dict=data/dict/vocab_wp.txt
  bpemodel=data/dict/bpemodel_unigram_1024

  lmdir=data/lm_train
  [ ! -d $lmdir ] && mkdir -p $lmdir

  # use external data
  train_sets="train_clean_100 train_clean_360 train_other_500"
  dev_sets="dev_clean dev_other"
  all_text=data/train_text
  (for f in $train_sets; do cat data_1kh/$f/text; done ) | sort -k1 > $all_text

  cat data/train_text | gzip -c > $lmdir/train_text.gz
  # combine external text and transcriptions and shuffle them with seed 777
  zcat $lm_data/librispeech-lm-norm.txt.gz $lmdir/train_text.gz |\
        spm_encode --model=${bpemodel}.model --output_format=piece > $lmdir/train.txt
  
  ( for f in $dev_sets; do cat data_1kh/$f/text; done ) | sort -k1 | cut -f 2- -d" " |\
        spm_encode --model=${bpemodel}.model --output_format=piece > $lmdir/valid.txt

  echo "[Stage 1] LM Preparation Finished."
fi


lm_exp=exp/libri_tfunilm16x512_4card_cosineanneal_ep20_maxlen120/

if [ $stage -le 2 ] && [ $end_stage -ge 2 ]; then

  [ ! -d $lm_exp ] && mkdir -p $lm_exp

  CUDA_VISIBLE_DEVICES="4,5,6,7" lm_train.py \
    --exp_dir $lm_exp \
    --train_config conf/lm.yaml \
    --data_config conf/lm_data.yaml \
    --lm_type "uniLM" \
    --batch_size 64 \
    --epochs 20 \
    --save_epoch 10 \
    --learning_rate 0.0001 \
    --end_patience 3 \
    --opt_type "cosine" \
    --weight_decay 0 \
    --print_freq 200 > $lm_exp/train.log 2>&1 &   #uncomment if you want to execute this in the backstage
 
  echo "[Stage 2] External LM Training Finished."
fi
