# Wav2Vec-U Tutorial
This tutorial is an elaboration of the original README for the Wav2Vec-U project conducted by Facebook Research. Some parts correspond to the same instructions they have provided and the rest are added to clarify more details.
## Installation Guide (To be Updated)
The Wav2Vec-U is implemented using the Fariseq toolkit and also utilize some other packages. 
There has been no specific python version mentioned but note that the repo uses faiss package for clustering which is supported by Python 3.6 and 3.7. Besides, if you want to use NVIDIA's apex for faster training you need to have a python version not older than 3.7. 

The list of dependecies and installation guides are discussed further. (The packages that need to be installed such as PyKaldi and Flashlight can be also be installed using Docker if you can and prefer to work with it, if you work on a system on which you have a sudo access you can also follow the path written in their dockerfiles. If non of these cases apply to you, you need to install these dependencies from the source. The Docker files are also a good resource for checking the dependencies and make sure you have them installed on your system).

### Vector Cluster

### Fairseq Installation
You can find the instruction to install Fariseq on its [github page](https://github.com/pytorch/fairseq). 
Note that the lastest version needs to be installed from the source. The package on PYPI is not updated and cause some incomaptibilities.
After installing Fairseq the envioronment variable FAIRSEQ_ROOT need to be set to the installation directory.
### rVADfast
rVADfast is used for activity voice detection and is used in audio preprocessing pipeline. You only need to clone the repository and set the RVAD_ROOT variable to that directory.
### KenLM
KenLM is a pckage for languag model inference. You can follow the [installation instruction](https://github.com/kpu/kenlm) and then set the KENLM_ROOT to the bin directory.
### PyKaldi
PyKaldi is the Python scripting layer for Kaldi toolkit. PyKaldi is used for decoding where a WFST is built. If you wish to install PyKaldi from the source there are some prerequisites that need to be satisfied (see [PyKaldi github page](https://github.com/pykaldi/pykaldi#installation)):
- Protobuf: Can be installed using the _install_protobuf.sh_ script in _pykaldi/tools_.
- Clif: Can be installed using the _install_clif.sh_ script in _pykaldi/tools_ or from the source. 
The prerequisites for Clif are listed [here](https://github.com/google/clif#installation). Using Ninja is recommended.
If you decide to install these packages from the source, remember to update the _CMAKE_PREFIX_PATH_.
- Kaldi: Can be installed in the _pykaldi/tools_ directory using the _install_kaldi.sh_ (set the _KALDI_ROOT_ to the kaldi installation directory)
Note that MKL is required for installing kaldi

### Flashlight
Flashlight: For decoding during inference [flashlight](https://github.com/flashlight/flashlight#building-and-installing) library is required. It has some prerequisites such as intel-opneai-mkl, KenLM, OpenBLAS, and FFTW3. 

## Training Guide
Here we elaborate more on the training steps that are explained on the [Wav2Veq-U github page](https://github.com/pytorch/fairseq/tree/main/examples/wav2vec/unsupervised). 
The training requires unpaired audio and text which will be preprocessed separately. 
  The wav2vec-U training procedure consists of three consecutive main steps:
* Preparation of speech representations and text data
* Generative adversarial training (GAN)
* Iterative self-training + Kaldi LM-decoding

### Audio preprocessing (for Librispeech)
Audio preprocessing starts with splitting the dataset, creating a manifest file, detecting silent segments and removing them using rVAD. 
```sh
# create a manifest file for the set original of audio files
python $FAIRSEQ_ROOT/examples/wav2vec/wav2vec_manifest.py /dir/to/save/audio/files --ext wav --dest /path/to/new/train.tsv --valid-percent 0

python scripts/vads.py -r $RVAD_ROOT < /path/to/train.tsv > train.vads

python scripts/remove_silence.py --tsv /path/to/train.tsv --vads train.vads --out /dir/to/save/audio/files

python $FAIRSEQ_ROOT/examples/wav2vec/wav2vec_manifest.py /dir/to/save/audio/files --ext wav --dest /path/to/new/train.tsv --valid-percent 0.01
```
Note that the silence removal should also be applied to the test set before the prepare_audio script is called. (The original silence removal scripts only consider train and valid sets)

The next steps are about capturing the audio representations using the wav2veq 2.0 model, applying PCA, detecting the segments using kmeans clustering and applying mean pooling. All these steps are followed in the prepare_audio.sh.

```sh
zsh scripts/prepare_audio.sh /dir/with/{train,test,valid}.tsv /output/dir /path/to/wav2vec2/model.pt 512 14
```
Note that if you have splits different than train/valid/test, you will need to modify this script. The last two arguments are the PCA dimensionality and the 0-based index of the layer from which to extract representations.

The major steps are as follows:
1. Extracting audio representation using wav2vec 2.0 (the pretrained models can be found on [Wav2Veq 2.0 github page](https://github.com/pytorch/fairseq/tree/main/examples/wav2vec#pre-trained-models)) 
Note that you need to download a chechpoint that has not been finetuned e.g. Wav2Vec 2.0 Large (LV-60).
2. Training the clustering model on the train set
3. Applying the clustering to all audio subsets
4. Training pca on the train set
5. Applying pca on all the audio subsets
6. Calculating the means of the PCA results
7. Applying mean pooling

The .wrd, .ltr, .phn files also need to be generated for the data subsets since they are used in evaluation and error rate calculation. 
- The .wrd files includes the real transcription of audio files in the order the are manifested in the {train, valid, test}.tsv after removing the silence (the manifest script reorders the files). 
- The .ltr files can be achieved using _wrd_to_ltr.py_:
   ```sh
    for split in ('train', 'valid', 'test'); do
        python $FAIRSEQ_ROOT/examples/wav2vec/unsupervised/scripts/wrd_to_ltr.py --compact < $target_dir/${split}.wrd > $target_dir/${split}.ltr
    done
    ```
- The .phn files can be generated using a phonemizer. (for English language G2P is the recommended phonemizer):
    ```sh
    for split in ('train', 'valid', 'test'); do
        python $FAIRSEQ_ROOT/examples/wav2vec/unsupervised/scripts/g2p_wrd_to_phn.py --compact < $target_dir/${split}.wrd > $target_dir/${split}.phn 
    done
    ```

### Text preprocessing (for Librispeech)
Text preprocessing is done through the prepare_text.sh script:
```sh
zsh scripts/prepare_text.sh language /path/to/text/file /output/dir 1000 espeak /path/to/fasttext/lid/model
```
The fourth argument is minimum number observations of phones to keep. If your text corpus is small, you might want to reduce this number.

The fifth argument is which phonemizer to use. Supported values are [espeak](http://espeak.sourceforge.net/), [espeak-ng](https://github.com/espeak-ng/espeak-ng), and [G2P](https://github.com/Kyubyong/g2p) (english only).

Pre-trained fasttext LID models can be downloaded [here](https://fasttext.cc/docs/en/language-identification.html).

The stpes in the script are as follows:
1. Normalizing and filtering text by removing numbers, punctuations, and words from other languages.
2. Fairseq preprocessing to generate the dict.txt file for words.
3. Applying the phonemizer and generating the phone.txt.
4. Generating the lexicon.lst.
5. Fairseq processing on the phonemes to filter the phonemes that are seen less than a threshold and generate phones/dict.txt.
6. Generating the filtered lexicon according to the phonems dict.
7. Inserting <SIL> into the phone transcription and generate lm.phones.filtered.txt.
8. Adding SIL to the phone dictionary.
9. Fairseq preprocessing using the updated phone dictionary.
10. Generating a 4-gram word lamguage model using kenlm and generating arpa and bin files.
11. Creating a fst decoding model using kaldi. (The outputs will be used for word decoding).
12. Generating 4-g phoneme language model using kenlm. (arpa and bin files).
13. Creating a fst decoding model using kaldi. (The outputs will be used for phoneme decoding).

### GAN Training
After the unpaired text and audio representations are prepared, they are used to to train the GAN model. The configuration file for GAN training can be found in the _config/gan_ directory. The keys with ??? value in the yml file are the arguments that should be determined by the user. 

We then use a GAN model to build a first unsupervised ASR model. The data preparation above of both speech features and text data is a necessary procedure that enables the generator to match speech to text in an unsupervised way. 

Launching GAN training on top of preprocessed features, with default hyperparameters can be done with:
```sh
PREFIX=w2v_unsup_gan_xp
TASK_DATA=/path/to/features/precompute_unfiltered_pca512_cls128_mean_pooled  
TEXT_DATA=/path/to/data/phones  # path to fairseq-preprocessed GAN data (phones dir)
KENLM_PATH=/path/to/data/phones/kenlm.phn.o4.bin  # KenLM 4-gram phoneme language model (LM data = GAN data here)

PYTHONPATH=$FAIRSEQ_ROOT PREFIX=$PREFIX fairseq-hydra-train \
    -m --config-dir config/gan \
    --config-name w2vu \
    task.data=${TASK_DATA} \
    task.text_data=${TEXT_DATA} \
    task.kenlm_path=${KENLM_PATH} \
    common.user_dir=${FAIRSEQ_ROOT}/examples/wav2vec/unsupervised \
    model.code_penalty=2 or 4 \
    model.gradient_penalty=1.5 or 2.0 \
    model.smoothness_weight=0.5 or 0.75 or 1.0 \
    common.seed=range(0,5)
```
#### Training path in fairseq toolkit
If you are curious to know how training the GAN is done using the Fairseq tool continue reading this section. 

Fairseq uses hydra, an open-source Python framework that simplifies the development of research and other complex applications. 
For better understanding the implementation the main training path is described as below:

The main scripts involved in training the model are fairseq_cli/train.py, fairseq/trainer.py, fairseq/tasks/fairseq_task.py, and /examples/wav2vec/unsupervised/tasks/unpaired_audio_text.pyunpaired_audio_text.py 

1. Training starts by calling fairseq_cli/hydra_train.py where the config is read from the correspondig config files and the arguments. In the _hydra_main(...) config details and the main(...) fucntion from fairseq_cli/train.py are passed to the distributed_utils.call_main(...) function found in (fairseq/distributed/utils.py).
2. call_main(...) determines if we are training on a single GPU or we are running distributed training and calls the relevent functions. We trained the GAN on a single GPU which directly calls for main(...) function in fairseq_cli/train.py.
3. The main(...) function in fairseq_cli/train.py does some preparation prior to training and also sets up the related task. 
* In our case, the task implementation can be found in the task directory of the repository (unpaired_audio_text.py). The unpaired_audio_text class inherits the FairseqTask class and overrides some of its parent's methods.
* Setting up the criterion, building the model (from models/w2vu.py), loading the valid set, and seting up the Trainer (fairseq/trainer.py) are other tasks that are done in this function. 
* The Trainer class is the main class for training. In the next steps of the training, loading checkpoint, loading dataset, setting up learning schedulers and many other tasks are managed by a Trainer object.
* The Trainer gets access to ab iterator on the train set through get_train_iterator(...) function which loads unpaired train audio and text dataset using task.load_dataset(...) with its implementation in the unpaired_audio_text.py. ExtractedFeaturesDataset(...) and data_utils.load_indexed_dataset(...) loads audio and text respectively.
* Training for an epoch takes place inside a while loop where the train(...) function is called.
4. The train function in fairseq_cli/train.py perform forward pass, backward pass, and weight updates using the Trainer's train_step(...).
* Validating the model and saving the checkpoints are then done in the validate_and_save(...). Note that the frequency of reporting validation results and saving the checkpoints are determined by the validate_interval and save_interval in the config file.
5. The whole training stops when the maximum number of updates (set as the optimization config) is reached (default is 150000).

## Inference Guide
Once we find the best checkpoint (chosen using unsupervised metric that combined language model perplexity and vocabulary usage), we can use it to generate phone labels (or word labels with an appropriate kaldi WFST):

```sh
python w2vu_generate.py --config-dir config/generate --config-name viterbi \
fairseq.common.user_dir=${FAIRSEQ_ROOT}/examples/wav2vec/unsupervised \
fairseq.task.data=/path/to/dir/with/features \
fairseq.common_eval.path=/path/to/gan/checkpoint \ 
fairseq.dataset.gen_subset=valid results_path=/where/to/save/transcriptions
```

- The decoding can be done either without a LM (e.g. Viterbi decoding) or with a LM (Kaldi Decoding) The decoding without LM works best on the same adjacent-mean-pooled features that the gan was trained on, while decoding with LM works better on features before the adjacent timestep mean-pooling step (without the "_pooled" suffix).

- The config for vitebi decoding to generate phone labels can be found in config/generate/viterbi.yaml.

- KALDI decoder can be applied before and after self-training for phoneme and word decoding. The full list of Config parameters can be found in the w2lu.generate.py. When you want to use the KALDI decoder, you should make sure that KALDI decoder config are included in the config file. You can find out more about the conifg in kaldi_decoder.py and kaldi_initializer.py. There two necessary config items:
    ```sh
    hlg_graph_path: path_to/HLG.phn.kenlm.wrd.o40003.fst (for word decoding) or path to HLG.phn.lm.phones.filtered.06.fst (for phoneme decoding)
    output_dict: path_to/kaldi_dict.kenlm.wrd.o40003.txt or path_to/kaldi_dict.lm.phones.filtered.06.txt 
    ```
-  The config for Kaldi decoding to generate phone labels can be found in config/generate/kaldi_phn.yaml
-  The config for Kaldi decoding to generate phone labels can be found in config/generate/kaldi_wrd.yaml
- Note that the targets argument indicates if the output are supoosed to be phonemes or words and in case of phonems (phn) the WER is actually the PER.
- Note that during inference the dict.phn.txt should be present in the audio features directory.

## Pretrained models
We have trained the model on LibriSpeech dataset. We used 100 hrs and 10 hrs of librispeech clean dataset (link) as the audio data, and used 10% of the librispeech language model corpus (link) as the text data. You can find the pretrained models here (here)
<table><>

## Self-training and Finetuning Guide
Coming soon...
## Depracations and troubleshooting
- capture_output is removed from python subprocess so the kaldi_init.py of fairseq needs to be updated. (replace with std_out = PIPE and std_err = PIPE where necessary)
- Kaldi add-self-loop-simple.cc might cause problem. If so remove the logging parts.
- Note that according to latest updates of fairseq the optimization amsgrad might cause problem and you might need to remove it from the config file.
