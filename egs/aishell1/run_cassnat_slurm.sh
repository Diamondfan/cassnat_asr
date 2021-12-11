#!/bin/bash
#SBATCH --job-name=cass_aishell
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

export MASTER_PORT=13422
export WORLD_SIZE=4
echo "NODELIST="${SLURM_NODELIST}
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

. cmd.sh
. path.sh

stage=1
end_stage=1

encoder_initial_model=exp/ar_conformer_baseline_interctc05_layer6_spect10m005f2m27_multistep05k21k90k/averaged.mdl
#asr_exp=exp/cassnat_multistep_initart_convenc_interctc_convdec_interce_expandl1r1/
asr_exp=exp/cassnat_multistep_initrand_convenc_interctc_convdec_interce_expandl1r1/

if [ $stage -le 1 ] && [ $end_stage -ge 1 ]; then

  if [ ! -d $asr_exp ]; then
    mkdir -p $asr_exp
  fi

  srun cassnat_train.py --exp_dir $asr_exp \
    --train_config conf/cassnat_train.yaml \
    --data_config conf/data.yaml \
    --epochs 120 \
    --save_epoch 40 \
    --end_patience 10 \
    --learning_rate 0.001 \
    --min_lr 0.00001 \
    --opt_type "multistep" \
    --weight_decay 0 \
    --label_smooth 0.1 \
    --ctc_alpha 0.5 \
    --interctc_alpha 0.5 \
    --interctc_layer 6 \
    --att_alpha 1 \
    --interce_alpha 0.01 \
    --interce_layer 6 \
    --use_cmvn \
    --seed 1234 \
    --use_slurm \
    --print_freq 50 > $asr_exp/train.log 2>&1 
    
    #--init_encoder \
    #--resume_model $encoder_initial_model \    

  echo "[Stage 1] ASR Training Finished."
fi

out_name='averaged.mdl'
if [ $stage -le 2 ] && [ $end_stage -ge 2 ]; then
  last_epoch=72
  
  average_checkpoints.py \
    --exp_dir $asr_exp \
    --out_name $out_name \
    --last_epoch $last_epoch \
    --num 12
  
  echo "[Stage 2] Average checkpoints Finished."

fi

if [ $stage -le 3 ] && [ $end_stage -ge 3 ]; then
  #exp=$asr_exp
  #exp=exp_cassnat/fanat_large_specaug_multistep_trig_src_initenc
  exp=exp/conv_fanat_convdec_maxlen4_interctc05_interce01_ce09_aftermapping

  #rnnlm_model=exp_cassnat/1kh_d512_multistep_ctc1_accum1_bth32_specaug/averaged.mdl
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
  nj=1
  batch_size=1
  test_set="test"

  for tset in $test_set; do
    echo "Decoding $tset..."
    desdir=$exp/${decode_type}_decode_average_bm1_${beam1}_sampdist_${s_dist}_samplenum_${s_num}_newlm${lmwt}_speech218_bth1_nj1/$tset/

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
        --rank_model 'at_baseline' \
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
