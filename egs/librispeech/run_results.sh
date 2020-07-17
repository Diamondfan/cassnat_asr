
#exp=exp/1kh_small_unigram_4card_ctc1_att1_schdler_accum1_gc5
exp=exp/1kh_small_lace_ctc1_att1_schdler_accum2_gc5/

for testset in dev_clean test_clean dev_other test_other; do
  for x in $exp/att*/$testset; do
    echo $x | sed 's:exp/.*/\([ac]\):\1:g'
    grep Sum/Avg $x/result.wrd.txt
  done
done
