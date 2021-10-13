
#exp=exp/1kh_d512_multistep_ctc1_accum1_bth32_specaug
#exp=exp/fanat_d512_multistep_specaug_notrig_src_uni_embed005_l2
#exp=exp/fanat_d512_multistep_specaug_notrig_src_uni_embed05_ce
#exp=exp/fanat_d512_multistep_specaug_notrig_src_uni_noebdloss
#exp=exp/fanat_d512_multistep_notrig_nosrc_nouni_ctc1_specaug_start5
exp=exp/fanat_large_specaug_multistep_trig_src_initenc
exp=exp/ar_convenc_e12d6_d256_multistep40k_160k_ctc1_accum1_bth32_specaugt2m40_warp/
exp=exp/conv_fanat_convdec_maxlen4_interctc05_interce01_ce09_aftermapping

for testset in dev test; do
  for x in $exp/*att*average*/$testset; do
    echo $x | sed 's:exp/.*/\([ac]\):\1:g'
    grep Sum/Avg $x/result.wrd.txt
  done
done
