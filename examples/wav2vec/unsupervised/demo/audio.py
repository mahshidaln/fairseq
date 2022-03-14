from ast import arg
import os
import sys
import math
import random
import os.path as osp
from tqdm import tqdm, trange
from pathlib import Path

from copy import deepcopy
from unittest import result
from scipy.signal import lfilter

import numpy as np
import soundfile as sf
import torchaudio
import torch
import faiss
import torch.nn.functional as F
from npy_append_array import NpyAppendArray


sys.path.insert(1, f'{os.environ["FAIRSEQ_ROOT"]}/examples/wav2vec/unsupervised/scripts')
from wav2vec_cluster_faiss import parse_faiss_specs
from wav2vec_cluster_faiss import Wav2VecFeatureReader as FaissFeatureReader
from wav2vec_extract_features import Wav2VecFeatureReader as WavFeatureReader

class AudioPreprocess:
    def __init__(self, input_files, target_dir, args):
        self.input_files = input_files
        self.target_dir = target_dir
        self.wav2vec_model = args.wav2vec_model
        self.layer = args.layer
        self.cluster_model = args.cluster_model
        self.pca_model = args.pca_model
        self.pca_batch = args.pca_batch
        self.pooling = args.pooling
        self.silent_files = []
        if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'

        self.mean_dir = osp.join(self.target_dir, 'mean')
        #self.pooled_dir = osp.join(self.target_dir, 'pooled')
        self.sil_dir = osp.join(self.target_dir, 'sil')
        self.feat_dir = osp.join(self.target_dir, 'feat')
        self.pca_dir = osp.join(self.target_dir, 'pca')
        os.makedirs(self.target_dir, exist_ok=True)
        os.makedirs(self.mean_dir, exist_ok=True)
        #os.makedirs(self.pooled_dir, exist_ok=True)
        os.makedirs(self.sil_dir, exist_ok=True)
        os.makedirs(self.feat_dir, exist_ok=True)
        os.makedirs(self.pca_dir, exist_ok=True)


    def rvad(self, speechproc, path):
        winlen, ovrlen, pre_coef, nfilter, nftt = 0.025, 0.01, 0.97, 20, 512
        ftThres = 0.5
        vadThres = 0.4
        opts = 1

        data, fs = sf.read(path)
        assert fs == 16_000, "sample rate must be 16khz"
        ft, flen, fsh10, nfr10 = speechproc.sflux(data, fs, winlen, ovrlen, nftt)

        # --spectral flatness --
        pv01 = np.zeros(ft.shape[0])
        pv01[np.less_equal(ft, ftThres)] = 1
        pitch = deepcopy(ft)

        pvblk = speechproc.pitchblockdetect(pv01, pitch, nfr10, opts)

        # --filtering--
        ENERGYFLOOR = np.exp(-50)
        b = np.array([0.9770, -0.9770])
        a = np.array([1.0000, -0.9540])
        fdata = lfilter(b, a, data, axis=0)

        # --pass 1--
        noise_samp, noise_seg, n_noise_samp = speechproc.snre_highenergy(
            fdata, nfr10, flen, fsh10, ENERGYFLOOR, pv01, pvblk
        )

        # sets noisy segments to zero
        for j in range(n_noise_samp):
            fdata[range(int(noise_samp[j, 0]), int(noise_samp[j, 1]) + 1)] = 0

        vad_seg = speechproc.snre_vad(
            fdata, nfr10, flen, fsh10, ENERGYFLOOR, pv01, pvblk, vadThres
        )
        return vad_seg, data


    def get_rvads(self):
        sys.path.append(os.environ.get('RVAD_ROOT'))
        import speechproc
        results = []
        stride = 160
        for fpath in tqdm(self.input_files):
            path = fpath
            vads, wav = self.rvad(speechproc, path)

            start = None
            vad_segs = []
            for i, v in enumerate(vads):
                if start is None and v == 1:
                    start = i * stride
                elif start is not None and v == 0:
                    vad_segs.append((start, i * stride))
                    start = None
            if start is not None:
                vad_segs.append((start, len(wav)))
            res = " ".join(f"{v[0]}:{v[1]}" for v in vad_segs)
            results.append(res)
        return results


    def remove_silence(self, vads):
        # load vads
        list_intervals = []
        for v in vads:
            interval = [
                [int(w.split(":")[0]), int(w.split(":")[1])] for w in v.rstrip().split()
            ]
            list_intervals.append(interval)

        # load audio and keep only intervals (i.e. remove silences)
        for i in trange(len(self.input_files)):
            data, _ = torchaudio.load(self.input_files[i])
            if len(list_intervals[i]) > 0:
                data_filtered = torch.cat(
                    [data[0][int(it[0]) : int(it[1])] for it in list_intervals[i]]
                ).unsqueeze(0)
            else:
                data_filtered = data

            outpath = self.sil_dir + "/" + "/".join(str(self.input_files[i]).split("/")[-1:])
            print(outpath)
            if not os.path.isdir("/".join(outpath.split("/")[:-1])):
                os.makedirs("/".join(outpath.split("/")[:-1]))
            if not os.path.exists(outpath):
                torchaudio.save(outpath, data_filtered, sample_rate=16000)
            else:
                print(outpath, "exists!")

            self.silent_files.append(Path(outpath))
      
    def extract_features(self):

        npaa = NpyAppendArray(osp.join(self.feat_dir, "trans.npy"))

        def get_iterator(files, checkpoint, layer):

            num = len(files)
            reader = WavFeatureReader(checkpoint, layer)

            def iterate():
                for fname in files:
                    w2v_feats = reader.get_feats(fname)
                    yield w2v_feats
            return iterate, num

        generator, num = get_iterator(self.silent_files, self.wav2vec_model, self.layer)
        iterator = generator()

        results = []
        for w2v_feats in tqdm(iterator, total=num):
            if len(w2v_feats) > 0:
                #results.append(w2v_feats.numpy())
                npaa.append(w2v_feats.numpy())
        return results

    def apply_cluster(self):
        spec = osp.basename(self.cluster_model)

        try:
            faiss_spec = parse_faiss_specs(spec.rstrip("/"))[0]
        except:
            print(spec)
            raise

        print("Faiss Spec:", faiss_spec, file=sys.stderr)

        if faiss_spec.pca:
            A = torch.from_numpy(np.load(osp.join(self.cluster_model, "pca_A.npy")))
            b = torch.from_numpy(np.load(osp.join(self.cluster_model, "pca_b.npy")))
            if(self.device == 'cuda'):
                A = A.cuda()
                b = b.cuda()
            print("Loaded PCA", file=sys.stderr)

        centroids = np.load(osp.join(self.cluster_model, "centroids.npy"))
        print("Loaded centroids", centroids.shape, file=sys.stderr)

        
        res = faiss.StandardGpuResources()
        index_flat = (
            faiss.IndexFlatL2(centroids.shape[1])
            if not faiss_spec.sphere
            else faiss.IndexFlatIP(centroids.shape[1])
        )
        
        faiss_index = faiss.index_cpu_to_gpu(res, 0, index_flat) if self.device =='cuda' else index_flat
        faiss_index.add(centroids)

        def get_iterator(files, checkpoint, layer):

            num = len(files)
            reader = FaissFeatureReader(checkpoint, layer)

            def iterate():
                for file in files:
                    feats = reader.get_feats(file)
                    fname = (str(file).split('/'))[-1]
                    yield feats.data, fname

            return iterate, num

        generator, num = get_iterator(self.silent_files, self.wav2vec_model, self.layer)
        iterator = generator()
        lengths = []
        results = []
        with torch.no_grad():
            for f, fname in tqdm(iterator, total=num):
                length = len(f)
                if faiss_spec.pca:
                    f = torch.mm(f, A) + b
                if faiss_spec.norm:
                    f = F.normalize(f, p=2, dim=-1)

                f = f.cpu().numpy()

                _, z = faiss_index.search(f, 1)
    
                res = [x.item() for x in z]
                results.append(res)
                lengths.append(length)
        
        return results, lengths


    def apply_pca(self, w2v_features):

        features = np.load(osp.join(self.feat_dir, "trans.npy"), mmap_mode="r")
        pca_A = torch.from_numpy(np.load(osp.join(self.pca_model, "512_pca_A.npy")))
        pca_b = torch.from_numpy(np.load(osp.join(self.pca_model, "512_pca_b.npy")))

        npaa = NpyAppendArray(osp.join(self.pca_dir, "trans.npy"))

        if(self.device == 'cuda'):
            pca_A = pca_A.cuda()
            pca_b = pca_b.cuda()

        batches = math.ceil(features.shape[0] / self.pca_batch)
        results = []
        with torch.no_grad():
            for b in trange(batches):
                start = b * self.pca_batch
                end = start + self.pca_batch
                x = torch.from_numpy(features[start:end])
                if(self.device == 'cuda'):
                    x = x.cuda()
                x = torch.matmul(x, pca_A) + pca_b
                res = x.cpu().numpy()
                #results.append(res)
                npaa.append(x.cpu().numpy())
        return results


    def merge_clusters(self, clusters, pcas, lengths):

        features = np.load(osp.join(self.pca_dir, "trans.npy"), mmap_mode="r")
        sizes = []
        offsets = []
        offset = 0
        
        for length in lengths:
            sizes.append(length)
            offsets.append(offset)
            offset += length

        npaa = NpyAppendArray(osp.join(self.mean_dir, "trans.npy"))
        length_file = osp.join(self.mean_dir, "trans.lengths")

        def merge(feats, clust):
            feats = torch.from_numpy(feats.copy())
            clust = torch.LongTensor(clust)
            _, counts = clust.unique_consecutive(return_counts=True)
            curr = 0

            merged = []
            for c in counts:
                c = c.item()
                start = curr
                end = curr + c
                curr += c
                if self.pooling == "mean":
                    new_x = feats[start:end].mean(dim=0)
                elif self.pooling == "sample":
                    new_x = feats[start + int(random.random() * c)]
                else:
                    raise NotImplementedError()
                merged.append(new_x)

            return torch.stack(merged, dim=0).numpy()
        
        results = []
        new_lengths = []

        with open(length_file, "w") as l_f:
            for size, offset, clust in tqdm(
                zip(sizes, offsets, clusters), total=len(sizes)
            ):
                end = size + offset
                feats = features[offset:end]
                feats = merge(feats, clust)
                new_lengths.append(len(feats))
                print(len(feats), file=l_f)
                results.append(feats)
                npaa.append(feats)
        
        return results, new_lengths

    def mean_pool(self, pca_mean, lengths, subsample_rate=0.5, remove_extra=False):

        features = pca_mean
        npaa = NpyAppendArray(osp.join(self.pooled_dir, "trans.npy"))
        length_file = osp.join(self.pooled_dir, "trans.lengths")

        fsz = features.shape[-1]
        start = 0
        results = []
        new_lengths = []
        with torch.no_grad():
            with open(length_file, "w") as lengths_out:
                for length in tqdm(lengths):
                    end = start + length
                    feats = features[start:end]
                    start += length
                    x = torch.from_numpy(feats)
                    if(self.device == 'cuda'):
                        x = x.cuda()
                    target_num = math.ceil(length * subsample_rate)
                    rem = length % target_num

                    if rem > 0:
                        if remove_extra:
                            to_rem = target_num - rem
                            target_num -= 1
                            x = x[:-to_rem]
                        else:
                            to_add = target_num - rem
                            x = F.pad(x, [0, 0, 0, to_add])
                            x[-to_add:] = x[-to_add - 1]

                    x = x.view(target_num, -1, fsz)
                    x = x.mean(dim=-2)
                    new_lengths.append(target_num)
                    results.append(x.cpu().numpy())
                    print(target_num, file=lengths_out)
                    npaa.append(x.cpu().numpy())

        return results, new_lengths

    def get_pca_mean(self):
        rvads = self.get_rvads()
        self.remove_silence(rvads)
        w2v_features = self.extract_features()
        clusters, lengths = self.apply_cluster()
        pcas = self.apply_pca(w2v_features)
        pca_mean, lengths = self.merge_clusters(clusters, pcas, lengths)
        #return pca_mean, lengths

    def get_pca_mean_pooled(self):
        pca_mean_pooled, lengths = self.mean_pool(self.get_pca_mean)
        return pca_mean_pooled, lengths
