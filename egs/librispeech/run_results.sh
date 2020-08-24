
#exp=exp/1kh_small_multistep_accum4_gc5_specaug_before_f30t40 #_ctc03
#exp=exp/1kh_small_multistep_accum1_gc5_specaug_before_f27t100
#exp=exp/1kh_small_unigram_4card_ctc1_att1_noamwarm_accum1_gc5_spec_aug
#exp=exp/fanat_multistep_trig_src_initenc_ctc1_sample_path
exp=exp/fanat_multistep_trig_src_initenc_ctc1_sample_path_SchD

for testset in dev_clean test_clean dev_other test_other; do
  for x in $exp/att*/$testset; do
    echo $x | sed 's:exp/.*/\([ac]\):\1:g'
    grep Sum/Avg $x/result.wrd.txt
  done
done
