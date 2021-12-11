
exp=exp/conv_fanat_best_interctc05_interce01_ce09_aftermapping/
search='ctc_att'  # check the specific decoding results

. path.sh
. utils/parse_options.sh

for testset in dev test; do
  for x in $exp/$search*/$testset; do
    echo $x | sed 's:exp/.*/\([ac]\):\1:g'
    grep Sum/Avg $x/result.wrd.txt
  done
done
