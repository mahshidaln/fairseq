from operator import sub
from pathlib import Path
from generate import gen_main

from audio import AudioPreprocess
from utils import load_config, gen_dirname


class English_Unsup:
    def __init__(self, input_files, target_dir, config_dir):
        self.audio_preprocess = AudioPreprocess(input_files, target_dir, load_config(f'{config_dir}/audio.yaml'))
        self.target_dir = target_dir
        self.config_dir = config_dir

    def prepare_audio(self):
        self.audio_preprocess.get_pca_mean()
        #pca_mean_pooled, length2 = self.audio_preprocess.get_pca_mean_pooled()

    def transcribe(self):
        self.prepare_audio()
        data_path = f'{self.target_dir}/mean'
        return gen_main(self.config_dir, 'kaldi_wrd', data_path)


def main():
    input_files = [Path('/h/malinoori/unsup/fairseq/examples/wav2vec/unsupervised/demo/data/61-70968-0000.flac'), Path('/h/malinoori/unsup/fairseq/examples/wav2vec/unsupervised/demo/data/2961-960-0000.flac')]
    target_dir = gen_dirname('./data')
    config_dir = './config'
    print(English_Unsup(input_files, target_dir, config_dir).transcribe())

if __name__=="__main__":
    main()
   