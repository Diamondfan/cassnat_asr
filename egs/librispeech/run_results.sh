
#repalce the dir with the experiment you want to check
exp=exp_cassnat/1kh_large_multistep_accum2_gc5_specaug_before_f30t40/
#exp=exp/1kh_e12d6_accum2_specaug_tmax4_multistep4k_30k_200k_disls_lr1e-3/
search='ctc_att'  # check the specific decoding results

for testset in test_clean test_other; do #dev_clean dev_other; do
  for x in $exp/$search*/$testset; do
    echo $x | sed 's:exp/.*/\([ac]\):\1:g'
    grep Sum/Avg $x/result.wrd.txt
  done
done
