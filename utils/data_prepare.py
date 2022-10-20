import soundfile as sf

dir = ['development', 'test'] #'dev_clean', 'dev_other','test_clean', 'test_other'] #, 'train_100h_sp']

for d in dir:
    print("Processing {}".format(d))
    wav_path = '/data/ruchao/workdir/cassnat_asr/egs/MyST/data/' + d + '/wav.scp'

    new_wav = '/data/ruchao/workdir/cassnat_asr/egs/MyST/data/' + d + '/wav_s.scp'


    with open(wav_path, 'r') as fin:
        for line in fin:
            cont = line.strip().split(' ')

            path = cont[-2]

            audio,_ = sf.read(path)

            #cont.append("blank")

            cont.append(str(len(audio)))

            line_n  = ' '.join(cont)

            line_n = line_n + "\n"

            with open(new_wav, 'a') as gin:
                gin.write(line_n)
