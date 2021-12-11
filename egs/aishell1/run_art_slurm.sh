#!/bin/bash
#SBATCH --job-name=art_aishell
#SBATCH --partition=v100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=128gb
#SBATCH --chdir=/home/mnt/ext/user/ruchao/workdir/E2EASR/egs/aishell1/
#SBATCH --output=//home/mnt/ext/user/ruchao/workdir/E2EASR/egs/aishell1/exp/ctc.out

### change 5-digit MASTER_PORT as you wish, slurm will raise Error if duplicated with others
### change WORLD_SIZE as gpus/node * num_nodes

export MASTER_PORT=12612
export WORLD_SIZE=4
echo "NODELIST="${SLURM_NODELIST}
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

data=/home/ruchao/Database/aishell/
lm_data=/home/ruchao/Database/LibriSpeech/libri_lm

stage=8
end_stage=8
featdir=data/fbank

. ./cmd.sh
. ./path.sh
. parse_options.sh

if [ $stage -le 1 ] && [ $end_stage -ge 1 ]; then

  local/aishell_data_prep.sh ${data}/data_aishell/wav ${data}/data_aishell/transcript
  # remove space in text
  for x in train dev test; do
    cp data/${x}/text data/${x}/text.org
    paste -d " " <(cut -f 1 -d" " data/${x}/text.org) <(cut -f 2- -d" " data/${x}/text.org | tr -d " ") \
        > data/${x}/text
    rm data/${x}/text.org
  done
  echo "[Stage 1] Data Preparation Finished."
fi

train_set=train_all
dev_set=dev
if [ $stage -le 2 ] && [ $end_stage -ge 2 ]; then
  for part in train dev test; do
    steps/make_fbank.sh --nj 16 --cmd $cmd --write_utt2num_frames true \
      data/$part exp/make_fbank/$part $featdir/$part
    utils/fix_data_dir.sh data/$part
  done
  
  # speed-perturbed
  utils/perturb_data_dir_speed.sh 0.9 data/train data/temp1
  utils/perturb_data_dir_speed.sh 1.0 data/train data/temp2
  utils/perturb_data_dir_speed.sh 1.1 data/train data/temp3
  utils/combine_data.sh --extra-files utt2uniq data/${train_set} data/temp1 data/temp2 data/temp3
  rm -r data/temp1 data/temp2 data/temp3

  steps/make_fbank.sh --cmd $cmd --nj 32 --write_utt2num_frames true \
    data/${train_set} exp/make_fbank/${train_set} $featdir/$part
  utils/fix_data_dir.sh data/${train_set}

  # compute global CMVN
  compute-cmvn-stats scp:data/${train_set}/feats.scp data/fbank/cmvn.ark || exit 1;
  echo "[Stage 2] Feature Extraction Finished"
fi

dict=data/dict/vocab_char.txt ; mkdir -p data/dict
if [ $stage -le 3 ] && [ $end_stage -ge 3 ]; then  
  echo "Create a dictionary..."
  text2token.py -s 1 -n 1 data/${train_set}/text | cut -f 2- -d" " | tr " " "\n" \
    | sort | uniq | grep -v -e '^\s*$' | awk '{print $0}' > ${dict}

  for part in ${train_set} dev test; do
    text2token.py -s 1 -n 1 data/$part/text > data/$part/token.scp
  done
  echo "[Stage 3] Dictionary and Transcription Finished."
fi

if [ $stage -le 4 ] && [ $end_stage -ge 4 ]; then
  echo "stage 4: LM Preparation"
  lmdir=data/lm_train
  if [ ! -d $lmdir ]; then
    mkdir -p $lmdir
  fi
  # use external data
  cat data/train_text | gzip -c > $lmdir/train_text.gz
  # combine external text and transcriptions and shuffle them with seed 777
  zcat $lm_data/librispeech-lm-norm.txt.gz $lmdir/train_text.gz |\
        spm_encode --model=${bpemodel}.model --output_format=piece > $lmdir/train.txt
  
  ( for f in $dev_set; do cat data/$f/text; done ) | sort -k1 | cut -f 2- -d" " |\
        spm_encode --model=${bpemodel}.model --output_format=piece > $lmdir/valid.txt

  echo "[Stage 4] LM Preparation Finished."
fi

if [ $stage -le 5 ] && [ $end_stage -ge 5 ]; then
  exp=exp/libri_tflm_unigram_4card_cosineanneal_ep10/
  if [ ! -d $exp ]; then
    mkdir -p $exp
  fi
  
  CUDA_VISIBLE_DEVICES="0,1,2,3" lm_train.py \
    --exp_dir $exp \
    --train_config conf/lm.yaml \
    --data_config conf/lm_data.yaml \
    --batch_size 64 \
    --epochs 10 \
    --save_epoch 3 \
    --anneal_lr_ratio 0.5 \
    --learning_rate 0.0001 \
    --min_lr 0.00001 \
    --patience 1 \
    --end_patience 5 \
    --opt_type "cosine" \
    --weight_decay 0 \
    --print_freq 200 > $exp/train.log 2>&1 &
 
  echo "[Stage 5] External LM Training Finished."
fi

exp=exp/ar_conformer_baseline_interctc05_layer6_spect10m005f2m27_multistep05k21k90k/

if [ $stage -le 6 ] && [ $end_stage -ge 6 ]; then
  if [ ! -d $exp ]; then
    mkdir -p $exp
  fi

  srun asr_train.py --exp_dir $exp \
    --train_config conf/transformer.yaml \
    --data_config conf/data.yaml \
    --epochs 100 \
    --save_epoch 40 \
    --learning_rate 0.001 \
    --min_lr 0.00001 \
    --end_patience 10 \
    --opt_type "multistep" \
    --weight_decay 0 \
    --label_smooth 0.1 \
    --ctc_alpha 0.5 \
    --interctc_alpha 0.5 \
    --interctc_layer 6 \
    --use_cmvn \
    --use_slurm \
    --seed 1234 \
    --print_freq 50 > $exp/train.log 2>&1
    
  echo "[Stage 6] ASR Training Finished."
fi

out_name='averaged.mdl'
if [ $stage -le 7 ] && [ $end_stage -ge 7 ]; then
  #exp=exp/1kh_d512_multistep_ctc1_accum1_bth32_specaug
  last_epoch=98
  
  average_checkpoints.py \
    --exp_dir $exp \
    --out_name $out_name \
    --last_epoch $last_epoch \
    --num 12
  
  #lm_exp=exp/libri_tflm_unigram_4card_cosineanneal_ep10/
  #last_epoch=9  
  
  #average_checkpoints.py \
  #  --exp_dir $lm_exp \
  #  --out_name $out_name \
  #  --last_epoch $last_epoch \
  #  --num 3

  echo "[Stage 7] Average checkpoints Finished."

fi

if [ $stage -le 8 ] && [ $end_stage -ge 8 ]; then
  exp=exp/ar_conformer_baseline_interctc05_layer6_spect10m005f2m27_multistep05k21k90k/

  test_model=$exp/$out_name
  rnnlm_model=exp/averaged_lm.mdl
  decode_type='ctc_att'
  attbeam=10 # check beam1 and beam2 in conf/decode.yaml, att beam
  ctcbeam=10 #20 # ctc beam
  lp=0
  ctcwt=0.4
  lmwt=0 #0.7
  nj=8
  batch_size=1
  test_set="dev test"

  for tset in $test_set; do
    echo "Decoding $tset..."
    desdir=$exp/${decode_type}_decode_nj1_ctc${ctcwt}_attbm_${attbeam}_ctcbm_${ctcbeam}_lp${lp}_newlmwt${lmwt}_lmeos/$tset/

    if [ ! -d $desdir ]; then
      mkdir -p $desdir
    fi
    
    split_scps=
    for n in $(seq $nj); do
      split_scps="$split_scps $desdir/feats.$n.scp"
    done
    utils/split_scp.pl data/$tset/feats.scp $split_scps || exit 1;
    
    $decode_cmd JOB=1:$nj $desdir/log/decode.JOB.log \
      CUDA_VISIBLE_DEVICES=JOB asr_decode.py \
        --test_config conf/decode.yaml \
        --lm_config conf/lm.yaml \
        --data_path $desdir/feats.JOB.scp \
        --resume_model $test_model \
        --result_file $desdir/token_results.JOB.txt \
        --batch_size $batch_size \
        --decode_type $decode_type \
        --ctc_weight $ctcwt \
        --rnnlm $rnnlm_model \
        --lm_weight $lmwt \
        --max_decode_ratio 0 \
        --use_cmvn \
        --print_freq 20 
    
    cat $desdir/token_results.*.txt | sort -k1,1 > $desdir/token_results.txt
    text2trn.py $desdir/token_results.txt $desdir/hyp.token.trn
    text2trn.py data/$tset/token.scp $desdir/ref.token.trn
 
    sclite -r $desdir/ref.token.trn -h $desdir/hyp.token.trn -i wsj -o all stdout > $desdir/result.wrd.txt
  done
  echo "[Stage 7] Decoding Finished."
fi


