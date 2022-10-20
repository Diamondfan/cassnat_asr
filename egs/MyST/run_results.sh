
#repalce the dir with the experiment you want to check
exp=exp/1kh_conformer_rel_maxlen20_e10d5_accum2_specaug_tmax10_multistep2k_40k_160k_ln #_shareff
search='ctc_att'  # check the specific decoding results

. path.sh
. utils/parse_options.sh

for testset in development test; do
  for x in $exp/$search*/$testset; do
    echo $x | sed 's:exp/.*/\([ac]\):\1:g'
    grep Sum/Avg $x/result.wrd.txt
  done
done
