
#repalce the dir with the experiment you want to check
exp=exp/1kh_large_multistep_accum2_gc5_specaug_before_f30t40/
search='ctc_att'  # check the specific decoding results

for testset in dev_clean test_clean dev_other test_other; do
  for x in $exp/$search*/$testset; do
    echo $x | sed 's:exp/.*/\([ac]\):\1:g'
    grep Sum/Avg $x/result.wrd.txt
  done
done
