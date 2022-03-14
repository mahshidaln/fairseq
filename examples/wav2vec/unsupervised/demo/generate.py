
"""
Run inference for pre-processed data with a trained model.
"""

import ast
from collections import namedtuple
from distutils.command.config import config
from dataclasses import dataclass, field
from enum import Enum, auto
from hydra.experimental import compose, initialize
from hydra.core.config_store import ConfigStore
import logging
import math
import os
#from hydra import compose, initialize
from omegaconf import OmegaConf
from typing import Optional
import sys

import editdistance
import torch

from hydra.core.hydra_config import HydraConfig

from fairseq import checkpoint_utils, progress_bar, tasks, utils
from fairseq.data.data_utils import post_process
from fairseq.dataclass.configs import FairseqDataclass, FairseqConfig
from fairseq.logging.meters import StopwatchMeter
from omegaconf import open_dict

from examples.speech_recognition.kaldi.kaldi_decoder import KaldiDecoderConfig

#logging.root.setLevel(logging.INFO)
#logging.basicConfig(stream=sys.stdout, level=logging.INFO)
#logger = logging.getLogger(__name__)


class DecoderType(Enum):
    VITERBI = auto()
    KENLM = auto()
    FAIRSEQ = auto()
    KALDI = auto()


@dataclass
class UnsupGenerateConfig(FairseqDataclass):
    fairseq: FairseqConfig = FairseqConfig()
    lm_weight: float = field(
        default=2.0,
        metadata={"help": "language model weight"},
    )
    w2l_decoder: DecoderType = field(
        default=DecoderType.VITERBI,
        metadata={"help": "type of decoder to use"},
    )
    kaldi_decoder_config: Optional[KaldiDecoderConfig] = None
    lexicon: Optional[str] = field(
        default=None,
        metadata={
            "help": "path to lexicon. This is also used to 'phonemize' for unsupvised param tuning"
        },
    )
    lm_model: Optional[str] = field(
        default=None,
        metadata={"help": "path to language model (kenlm or fairseq)"},
    )
    kenlm_model: Optional[str] = field(
        default=None,
        metadata={"help": "path to language model (kenlm or fairseq)"},
    )
    unit_lm: bool = field(
        default=False,
        metadata={"help": "whether to use unit lm"},
    )
    beam_threshold: float = field(
        default=50.0,
        metadata={"help": "beam score threshold"},
    )
    beam_size_token: float = field(
        default=100.0,
        metadata={"help": "max tokens per beam"},
    )
    beam: int = field(
        default=5,
        metadata={"help": "decoder beam size"},
    )
    nbest: int = field(
        default=1,
        metadata={"help": "number of results to return"},
    )
    word_score: float = field(
        default=1.0,
        metadata={"help": "word score to add at end of word"},
    )
    unk_weight: float = field(
        default=-math.inf,
        metadata={"help": "unknown token weight"},
    )
    sil_weight: float = field(
        default=0.0,
        metadata={"help": "silence token weight"},
    )
    targets: Optional[str] = field(
        default=None,
        metadata={"help": "extension of ground truth labels to compute UER"},
    )
    results_path: Optional[str] = field(
        default=None,
        metadata={"help": "where to store results"},
    )
    post_process: Optional[str] = field(
        default=None,
        metadata={"help": "how to post process results"},
    )
    vocab_usage_power: float = field(
        default=2,
        metadata={"help": "for unsupervised param tuning"},
    )

    viterbi_transcript: Optional[str] = field(
        default=None,
        metadata={"help": "for unsupervised param tuning"},
    )
    min_lm_ppl: float = field(
        default=0,
        metadata={"help": "for unsupervised param tuning"},
    )
    min_vt_uer: float = field(
        default=0,
        metadata={"help": "for unsupervised param tuning"},
    )

    blank_weight: float = field(
        default=0,
        metadata={"help": "value to add or set for blank emission"},
    )
    blank_mode: str = field(
        default="set",
        metadata={
            "help": "can be add or set, how to modify blank emission with blank weight"
        },
    )
    sil_is_blank: bool = field(
        default=False,
        metadata={"help": "if true, <SIL> token is same as blank token"},
    )

    unsupervised_tuning: bool = field(
        default=False,
        metadata={
            "help": "if true, returns a score based on unsupervised param selection metric instead of UER"
        },
    )
    is_ax: bool = field(
        default=False,
        metadata={
            "help": "if true, assumes we are using ax for tuning and returns a tuple for ax to consume"
        },
    )
    


def get_dataset_itr(cfg, task):
    return task.get_batch_iterator(
        dataset=task.dataset(cfg.fairseq.dataset.gen_subset),
        max_tokens=cfg.fairseq.dataset.max_tokens,
        max_sentences=cfg.fairseq.dataset.batch_size,
        max_positions=(sys.maxsize, sys.maxsize),
        ignore_invalid_inputs=cfg.fairseq.dataset.skip_invalid_size_inputs_valid_test,
        required_batch_size_multiple=cfg.fairseq.dataset.required_batch_size_multiple,
        num_shards=cfg.fairseq.dataset.num_shards,
        shard_id=cfg.fairseq.dataset.shard_id,
        num_workers=cfg.fairseq.dataset.num_workers,
        data_buffer_size=cfg.fairseq.dataset.data_buffer_size,
    ).next_epoch_itr(shuffle=False)


def process_predictions(
    cfg: UnsupGenerateConfig,
    hypos,
    tgt_dict,
):

    for i, hypo in enumerate(hypos[: min(len(hypos), cfg.nbest)]):
        if torch.is_tensor(hypo["tokens"]):
            tokens = hypo["tokens"].int().cpu()
            tokens = tokens[tokens >= tgt_dict.nspecial]
            hyp_pieces = tgt_dict.string(tokens)
        else:
            hyp_pieces = " ".join(hypo["tokens"])

        if "words" in hypo and len(hypo["words"]) > 0:
            hyp_words = " ".join(hypo["words"])
        else:
            hyp_words = post_process(hyp_pieces, cfg.post_process)
    return hyp_words


def optimize_models(cfg: UnsupGenerateConfig, use_cuda, models):
    """Optimize ensemble for generation"""
    for model in models:
        model.eval()
        if cfg.fairseq.common.fp16:
            model.half()
        if use_cuda:
            model.cuda()

def generate(cfg: UnsupGenerateConfig, models, saved_cfg, use_cuda):
    task = tasks.setup_task(cfg.fairseq.task)
    saved_cfg.task.labels = cfg.fairseq.task.labels
    task.load_dataset(cfg.fairseq.dataset.gen_subset, task_cfg=saved_cfg.task)
    # Set dictionary
    tgt_dict = task.target_dictionary
    #logger.info(
    #    "| {} {} {} examples".format(
    #        cfg.fairseq.task.data,
    #        cfg.fairseq.dataset.gen_subset,
    #        len(task.dataset(cfg.fairseq.dataset.gen_subset)),
    #    )
    #)
    # Load dataset (possibly sharded)
    itr = get_dataset_itr(cfg, task)
    # Initialize generator
    gen_timer = StopwatchMeter()

    def build_generator(cfg: UnsupGenerateConfig):
        w2l_decoder = cfg.w2l_decoder
        if w2l_decoder == DecoderType.VITERBI:
            from examples.speech_recognition.w2l_decoder import W2lViterbiDecoder

            return W2lViterbiDecoder(cfg, task.target_dictionary)
        elif w2l_decoder == DecoderType.KENLM:
            from examples.speech_recognition.w2l_decoder import W2lKenLMDecoder

            return W2lKenLMDecoder(cfg, task.target_dictionary)
        elif w2l_decoder == DecoderType.FAIRSEQ:
            from examples.speech_recognition.w2l_decoder import W2lFairseqLMDecoder

            return W2lFairseqLMDecoder(cfg, task.target_dictionary)
        elif w2l_decoder == DecoderType.KALDI:
            from examples.speech_recognition.kaldi.kaldi_decoder import KaldiDecoder

            assert cfg.kaldi_decoder_config is not None

            return KaldiDecoder(
                cfg.kaldi_decoder_config,
                cfg.beam,
            )
        else:
            raise NotImplementedError(
                "only wav2letter decoders with (viterbi, kenlm, fairseqlm) options are supported at the moment but found "
                + str(w2l_decoder)
            )

    generator = build_generator(cfg)

    kenlm = None
    fairseq_lm = None
    if cfg.lm_model is not None:
        import kenlm

        kenlm = kenlm.Model(cfg.lm_model)

    num_sentences = 0
    if cfg.results_path is not None and not os.path.exists(cfg.results_path):
        os.makedirs(cfg.results_path)

    errs_t = 0
    lengths_hyp_t = 0
    lengths_hyp_unit_t = 0
    lengths_t = 0
    count = 0
    num_feats = 0
    all_hyp_pieces = []
    all_hyp_words = []

    gen_timer.start()

    start = 0
    end = len(itr)

    hypo_futures = None
    if cfg.w2l_decoder == DecoderType.KALDI:
        #logger.info("Extracting features")
        hypo_futures = []
        samples = []
        with progress_bar.build_progress_bar(cfg.fairseq.common, itr) as t:
            for i, sample in enumerate(t):
                if "net_input" not in sample or i < start or i >= end:
                    continue
                if "padding_mask" not in sample["net_input"]:
                    sample["net_input"]["padding_mask"] = None

                hypos, num_feats = gen_hypos(
                    generator, models, num_feats, sample, task, use_cuda
                )
                hypo_futures.append(hypos)
                samples.append(sample)
        itr = list(zip(hypo_futures, samples))
        start = 0
        end = len(itr)
        #logger.info("Finished extracting features")

    with progress_bar.build_progress_bar(cfg.fairseq.common, itr) as t:
        for i, sample in enumerate(t):
            if i < start or i >= end:
                continue

            if hypo_futures is not None:
                hypos, sample = sample
                hypos = [h.result() for h in hypos]
            else:
                if "net_input" not in sample:
                    continue

                hypos, num_feats = gen_hypos(
                    generator, models, num_feats, sample, task, use_cuda
                )

            for i, sample_id in enumerate(sample["id"].tolist()):

                # Process top predictions
                hyp_words = process_predictions(
                    cfg,
                    hypos[i],
                    tgt_dict,
                )
                count += 1
                all_hyp_words.append(hyp_words)

            num_sentences += (
                sample["nsentences"] if "nsentences" in sample else sample["id"].numel()
            )

    gen_timer.stop(lengths_hyp_t)
    return all_hyp_words


def gen_hypos(generator, models, num_feats, sample, task, use_cuda):
    sample = utils.move_to_cuda(sample) if use_cuda else sample

    if "features" in sample["net_input"]:
        sample["net_input"]["dense_x_only"] = True
        num_feats += (
            sample["net_input"]["features"].shape[0]
            * sample["net_input"]["features"].shape[1]
        )
    hypos = task.inference_step(generator, models, sample, None)
    return hypos, num_feats


def main(cfg: UnsupGenerateConfig, model=None):
    if (
        cfg.fairseq.dataset.max_tokens is None
        and cfg.fairseq.dataset.batch_size is None
    ):
        cfg.fairseq.dataset.max_tokens = 1024000

    use_cuda = torch.cuda.is_available() and not cfg.fairseq.common.cpu

    task = tasks.setup_task(cfg.fairseq.task)

    overrides = ast.literal_eval(cfg.fairseq.common_eval.model_overrides)

    if cfg.fairseq.task._name == "unpaired_audio_text":
        overrides["model"] = {
            "blank_weight": cfg.blank_weight,
            "blank_mode": cfg.blank_mode,
            "blank_is_sil": cfg.sil_is_blank,
            "no_softmax": True,
            "segmentation": {
                "type": "NONE",
            },
        }
    else:
        overrides["model"] = {
            "blank_weight": cfg.blank_weight,
            "blank_mode": cfg.blank_mode,
        }

    if model is None:
        # Load ensemble
        #logger.info("| loading model(s) from {}".format(cfg.fairseq.common_eval.path))
        models, saved_cfg = checkpoint_utils.load_model_ensemble(
            cfg.fairseq.common_eval.path.split("\\"),
            arg_overrides=overrides,
            task=task,
            suffix=cfg.fairseq.checkpoint.checkpoint_suffix,
            strict=(cfg.fairseq.checkpoint.checkpoint_shard_count == 1),
            num_shards=cfg.fairseq.checkpoint.checkpoint_shard_count,
        )
        optimize_models(cfg, use_cuda, models)
    else:
        models = [model]
        saved_cfg = cfg.fairseq

    with open_dict(saved_cfg.task):
        saved_cfg.task.shuffle = False
        saved_cfg.task.sort_by_length = False

    gen_result = generate(cfg, models, saved_cfg, use_cuda)
    return gen_result

def gen_main(cfg_path, cfg_name, data):
    cs = ConfigStore.instance()
    cs.store(name=cfg_name, node=UnsupGenerateConfig)
    initialize(config_path=cfg_path, job_name="unsup_gen")
    cfg = compose(config_name=cfg_name)
    cfg.fairseq.task.data = data
    cfg = OmegaConf.create(
        OmegaConf.to_container(cfg, resolve=False, enum_to_str=False)
    )
    OmegaConf.set_struct(cfg, True)
    utils.import_user_module(cfg.fairseq.common)
    res = main(cfg)
    return res
