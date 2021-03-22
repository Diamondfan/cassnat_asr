
#exp=exp/1kh_d512_multistep_ctc1_accum1_bth32_specaug
#exp=exp/fanat_d512_multistep_specaug_notrig_src_uni_embed005_l2
#exp=exp/fanat_d512_multistep_specaug_notrig_src_uni_embed05_ce
#exp=exp/fanat_d512_multistep_specaug_notrig_src_uni_noebdloss
#exp=exp/fanat_d512_multistep_notrig_nosrc_nouni_ctc1_specaug_start5
#exp=exp/fanat_large_specaug_multistep_trig_src_initenc

for testset in dev test; do
  for x in $exp/*att*average*/$testset; do
    echo $x | sed 's:exp/.*/\([ac]\):\1:g'
    grep Sum/Avg $x/result.wrd.txt
  done
done
