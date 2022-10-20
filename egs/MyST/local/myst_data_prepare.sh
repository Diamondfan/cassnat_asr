#!/usr/bin/env bash


src=$1
dst=$2

# all utterances are FLAC compressed
if ! which flac >&/dev/null; then
   echo "Please install 'flac' on ALL worker nodes!"
   exit 1
fi


mkdir -p $dst || exit 1;
mkdir -p ${dst}_wotrn || exit 1;
[ ! -d $src ] && echo "$0: no such directory $src" && exit 1;


wav_scp=$dst/wav.scp; [[ -f "$wav_scp" ]] && rm $wav_scp
wav_scp_notrn=${dst}_wotrn/wav.scp; [[ -f "$wav_scp_notrn" ]] && rm $wav_scp_notrn
trans=$dst/text; [[ -f "$trans" ]] && rm $trans
utt2spk=$dst/utt2spk; [[ -f "$utt2spk" ]] && rm $utt2spk
utt2spk_notrn=${dst}_wotrn/utt2spk; [[ -f "$utt2spk_notrn" ]] && rm $utt2spk_notrn
tmpdir=tmp
mkdir -p $tmpdir

for speaker_dir in $(find -L $src -mindepth 1 -maxdepth 1 -type d | sort); do
  speaker=$(basename $speaker_dir)

  for session_dir in $(find -L $speaker_dir/ -mindepth 1 -maxdepth 1 -type d | sort); do
    session=$(basename $session_dir)

    find -L $session_dir/ -iname "*.trn" | sort | xargs -I% basename % .trn > $tmpdir/utt
    ntrn=$(wc -l < $tmpdir/utt)
 
    find -L $session_dir/ -iname "*.flac" | sort | xargs -I% basename % .flac | \
      awk -v "dir=$session_dir" '{printf "%s flac -c -d -s %s/%s.flac |\n", $0, dir, $0}' > $tmpdir/wav
    
    awk -v "speaker=$speaker" -v "session=$session" '{printf "%s myst_%s\n", $1, speaker}' \
      <$tmpdir/wav > $tmpdir/utt2spk || exit 1
 
    if [ $ntrn -eq 0 ]; then
      cat $tmpdir/wav >> $wav_scp_notrn
      cat $tmpdir/utt2spk >> $utt2spk_notrn
    else
      cat $tmpdir/wav >> $wav_scp
      cat $tmpdir/utt2spk >> $utt2spk

      find -L $session_dir/ -iname "*.trn" | sort > $tmpdir/trn
      while read line; do
        [ -f $line ] || error_exit "Cannot find transcription file '$line'";
        head -n1 "$line" | sed "s:<.*>::g" | sed "s:(\*)::g" | sed "s:(())::g" | sed "s:+::g" | sed "s:\*::g" | sed "s:[()]::g" | sed "s:  : :g" | sed "s:  : :g" | sed "s:^ ::g" | sed "s/\xC2\xA0/ /g"| tr '[:lower:]' '[:upper:]'
      done < $tmpdir/trn > $tmpdir/trans
      paste -d" " $tmpdir/utt $tmpdir/trans | sort >> $trans
    fi
  done
  echo "Done speaker $speaker!"
done
rm -rf $tmpdir

spk2utt=$dst/spk2utt
utils/utt2spk_to_spk2utt.pl <$utt2spk >$spk2utt || exit 1

ntrans=$(wc -l <$trans)
nutt2spk=$(wc -l <$utt2spk)
! [ "$ntrans" -eq "$nutt2spk" ] && \
  echo "Inconsistent #transcripts($ntrans) and #utt2spk($nutt2spk)" && exit 1;

utils/validate_data_dir.sh --no-feats --no-text $dst || exit 1;

echo "$0: successfully prepared data in $dst"

exit 0
