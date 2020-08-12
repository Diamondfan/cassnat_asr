
#exp=exp/1kh_small_unigram_4card_ctc1_att1_noamwarm_accum1_gc5
#exp=exp/fanat_trig_src_initenc_ctc1_endctc/
#exp=exp/fanat_multistep_trig_src_initenc_ctc1_endctc
exp=exp/fanat_large_specaug_multistep_trig_src_initenc
#exp=exp/fanat_multistep_trig_src_initenc_ctc1_usekd/

for testset in dev_clean test_clean dev_other test_other; do
  for x in $exp/ctc_att*average*/$testset; do
    echo $x | sed 's:exp/.*/\([ac]\):\1:g'
    grep Sum/Avg $x/result.wrd.txt
  done
done
