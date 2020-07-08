
exp=exp/1kh_small_drp01_l0r2_lda03_ls01_adam_lr2e-4_dc10_nd4
#exp=exp/1kh_small_drp01_l0r2_lda03_ls01_adam/
#exp=exp/1kh_big_drp01_l0r2_lda03_ls01_adam_lr2e-4_dc10_nd4
data=test_clean

for testset in dev_clean test_clean dev_other test_other; do
  for x in $exp/ctc_only*/$testset; do
    echo $x | sed 's:exp/.*/\([ac]\):\1:g'
    grep Sum/Avg $x/result.wrd.txt
  done
done
