
#exp=exp/fanat_unigram_4gpu_noamwarm_accum1_trig_nosrc_noembed
#exp=exp/1kh_small_unigram_4card_ctc1_att1_multistep_accum1_gc5_spec_aug_first/
#exp=exp/1kh_small_unigram_4card_ctc1_att1_noamwarm_accum1_gc5_spec_aug
exp=exp/fanat_multistep_trig_src_initenc_ctc1

for testset in dev_clean test_clean dev_other test_other; do
  for x in $exp/*/$testset; do
    echo $x | sed 's:exp/.*/\([ac]\):\1:g'
    grep Sum/Avg $x/result.wrd.txt
  done
done
