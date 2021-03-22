
#repalce the dir with the experiment you want to check
#exp=exp_cassnat/1kh_large_multistep_accum2_gc5_specaug_before_f30t40/
#exp=exp_cassnat/fanat_large_specaug_multistep_trig_src_initenc_SchD_shift_path0
#exp=exp/1kh_conformer_abs_e12d6_accum2_specaug_tmax4_multistep2k_40k_160k_ln/
exp=exp/1kh_conformer_rel_maxlen20_e10d5_accum2_specaug_tmax10_multistep2k_40k_160k_ln #_shareff
#exp=exp/1kh_conformer_baseline_interctc05/
#exp=exp/conv_fanat_e10m3d3_max_specaug_multistep_initenc_no_convdec/
#exp=exp/conv_fanat_e10m2d4_max_specaug_multistep_initenc_convdec_maxlen4_kernel7 #_ctxtrig1/ #_topk4/ #_nosqrt/
#exp=exp_cassnat/fanat_large_specaug_multistep_trig_src_initenc_SchD_topk_path4/
search='ctc_att' #'att_only' #'ctc_att'  # check the specific decoding results

for testset in dev_clean test_clean dev_other test_other; do
  for x in $exp/$search*/$testset; do
    echo $x | sed 's:exp/.*/\([ac]\):\1:g'
    grep Sum/Avg $x/result.wrd.txt
  done
done
