
exp=exp/tri6b
data=test_clean
lm=fglarge

for testset in dev_clean test_clean dev_other test_other;
do
  grep WER $exp/decode_${lm}_$testset/wer_* | utils/best_wer.sh
done
