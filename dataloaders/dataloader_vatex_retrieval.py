from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import os
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from collections import defaultdict
import json
import random
import sys
sys.path.append('..')
from dataloaders.rawvideo_util import RawVideoExtractor
from dataloaders.rawframes_util import RawFramesExtractor


class VATEX_DataLoader(Dataset):
    """VATEX dataset loader."""
    def __init__(
            self,
            subset,
            data_path,
            features_path,
            tokenizer,
            max_words=30,
            feature_framerate=1.0,
            max_frames=100,
            image_resolution=224,
            frame_order=0,
            slice_framepos=0,
            strategy=1
    ):
        self.subset = subset
        assert self.subset in ["train", "val", "test"]

        self.data_path = data_path
        self.features_path = os.path.join(features_path, self.subset + '_video')
        self.feature_framerate = feature_framerate
        self.max_words = max_words
        self.max_frames = max_frames
        self.tokenizer = tokenizer

        self.frame_order = frame_order
        assert self.frame_order in [0, 1, 2]

        self.slice_framepos = slice_framepos
        self.strategy = strategy
        assert self.slice_framepos in [0, 1, 2]

        video_id_path_dict = {}
        video_id_path_dict["train"] = os.path.join(self.data_path, 'vatex_data', "train_list.txt")
        video_id_path_dict["val"] = os.path.join(self.data_path, 'vatex_data', "val_list.txt")
        video_id_path_dict["test"] = os.path.join(self.data_path, 'vatex_data', "test_list.txt")
        caption_file = os.path.join(self.data_path, 'vatex_data', "vatex_data.json")

        with open(video_id_path_dict[self.subset], 'r') as fp:
            video_ids = [itm.strip() for itm in fp.readlines()]
        

        self.caption = json.load(open(caption_file, 'r'))

        video_dict = {}
        for root, dub_dir, video_files in os.walk(self.features_path):

            for video_file in dub_dir: # frames----------
                video_id_ = video_file # frames----------


                if video_id_ not in video_ids:
                    continue
                file_path_ = os.path.join(root, video_file)
                video_dict[video_id_] = file_path_

        self.video_dict = video_dict

        self.sample_len = 0
        self.sentences_dict = {}
        self.cut_off_points = []
        for video_id in video_ids:
            assert video_id in self.caption
            if video_id in self.video_dict:
                for cap in self.caption[video_id]['enCap']:
                    self.sentences_dict[len(self.sentences_dict)] = (video_id, cap)
                self.cut_off_points.append(len(self.sentences_dict))

        ## below variables are used to multi-sentences retrieval
        # self.cut_off_points: used to tag the label when calculate the metric
        # self.sentence_num: used to cut the sentence representation
        # self.video_num: used to cut the video representation
        self.multi_sentence_per_video = True    # !!! important tag for eval
        if self.subset == "val" or self.subset == "test":
            self.sentence_num = len(self.sentences_dict)
            self.video_num = len(self.video_dict)
            assert len(self.cut_off_points) == self.video_num
            print("For {}, sentence number: {}".format(self.subset, self.sentence_num))
            print("For {}, video number: {}".format(self.subset, self.video_num))

        print("Video number: {}".format(len(self.video_dict)))
        print("Total Paire: {}".format(len(self.sentences_dict)))


        if self.subset == 'test':
            dict_path = 'data/vatex_test_vitb32_it30_prompt.json'
            self.generate_caps_dict = json.load(open(dict_path, 'r'))
        elif self.subset == 'train':
            dict_path = 'data/vatex_train_vitb32_max1_nopro_final.json'
            self.generate_caps_dict = json.load(open(dict_path, 'r'))

        if self.subset == 'train' :
            for video_id in self.generate_caps_dict.keys():
                if video_id not in video_ids:
                    continue
                self.sentences_dict[len(self.sentences_dict)] = (video_id,self.generate_caps_dict[video_id]['max1'])

        self.sample_len = len(self.sentences_dict)
        self.rawVideoExtractor = RawVideoExtractor(framerate=feature_framerate, size=image_resolution)
        print('strategy====',self.strategy)
        self.rawFramesExtractor = RawFramesExtractor(
            num_segments=max_frames, size=image_resolution, random_shift=False, strategy=self.strategy if self.subset=='train' else 4)
                    
        self.SPECIAL_TOKEN = {"CLS_TOKEN": "<|startoftext|>", "SEP_TOKEN": "<|endoftext|>",
                              "MASK_TOKEN": "[MASK]", "UNK_TOKEN": "[UNK]", "PAD_TOKEN": "[PAD]"}


    def __len__(self):
        return self.sample_len

    def _get_text(self, video_id, caption):
        k = 1
        choice_video_ids = [video_id]
        pairs_text = np.zeros((k, self.max_words), dtype=np.long)
        pairs_mask = np.zeros((k, self.max_words), dtype=np.long)
        pairs_segment = np.zeros((k, self.max_words), dtype=np.long)

        for i, video_id in enumerate(choice_video_ids):
            words = self.tokenizer.tokenize(caption)

            words = [self.SPECIAL_TOKEN["CLS_TOKEN"]] + words
            total_length_with_CLS = self.max_words - 1
            if len(words) > total_length_with_CLS:
                words = words[:total_length_with_CLS]
            words = words + [self.SPECIAL_TOKEN["SEP_TOKEN"]]

            input_ids = self.tokenizer.convert_tokens_to_ids(words)
            input_mask = [1] * len(input_ids)
            segment_ids = [0] * len(input_ids)
            while len(input_ids) < self.max_words:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)
            assert len(input_ids) == self.max_words
            assert len(input_mask) == self.max_words
            assert len(segment_ids) == self.max_words

            pairs_text[i] = np.array(input_ids)
            pairs_mask[i] = np.array(input_mask)
            pairs_segment[i] = np.array(segment_ids)

        return pairs_text, pairs_mask, pairs_segment, choice_video_ids

    def _get_rawframes(self, choice_video_ids):

        video_mask = np.zeros((len(choice_video_ids), self.max_frames), dtype=np.long)
        max_video_length = [0] * len(choice_video_ids)

        # Pair x L x T x 3 x H x W
        video = np.zeros((len(choice_video_ids), self.max_frames, 1, 3,
                          self.rawFramesExtractor.size, self.rawFramesExtractor.size), dtype=np.float)


        for i, video_id in enumerate(choice_video_ids):
            video_path = self.video_dict[video_id]

            raw_video_data = self.rawFramesExtractor.get_video_data(video_path)
            raw_video_data = raw_video_data['video']

            if len(raw_video_data.shape) > 3:
                # L x T x 3 x H x W
                raw_video_slice = self.rawFramesExtractor.process_raw_data(raw_video_data)
                if self.max_frames < raw_video_slice.shape[0]:
                    if self.slice_framepos == 0:
                        video_slice = raw_video_slice[:self.max_frames, ...]
                    elif self.slice_framepos == 1:
                        video_slice = raw_video_slice[-self.max_frames:, ...]
                    else:
                        sample_indx = np.linspace(0, raw_video_slice.shape[0] - 1, num=self.max_frames, dtype=int)
                        video_slice = raw_video_slice[sample_indx, ...]
                else:
                    video_slice = raw_video_slice

                video_slice = self.rawFramesExtractor.process_frame_order(video_slice, frame_order=self.frame_order)

                slice_len = video_slice.shape[0]
                max_video_length[i] = max_video_length[i] if max_video_length[i] > slice_len else slice_len
                if slice_len < 1:
                    pass
                else:
                    video[i][:slice_len, ...] = video_slice
            else:
                print("video path: {} error. video id: {}".format(video_path, video_id))

        for i, v_length in enumerate(max_video_length):
            video_mask[i][:v_length] = [1] * v_length

        return video, video_mask


    def _get_rawvideo(self, choice_video_ids):

        video_mask = np.zeros((len(choice_video_ids), self.max_frames), dtype=np.long)
        max_video_length = [0] * len(choice_video_ids)

        # Pair x L x T x 3 x H x W
        video = np.zeros((len(choice_video_ids), self.max_frames, 1, 3,
                          self.rawVideoExtractor.size, self.rawVideoExtractor.size), dtype=np.float)

        for i, video_id in enumerate(choice_video_ids):
            video_path = self.video_dict[video_id]

            raw_video_data = self.rawVideoExtractor.get_video_data(video_path)
            raw_video_data = raw_video_data['video']

            if len(raw_video_data.shape) > 3:
                raw_video_data_clip = raw_video_data
                # L x T x 3 x H x W
                raw_video_slice = self.rawVideoExtractor.process_raw_data(raw_video_data_clip)
                if self.max_frames < raw_video_slice.shape[0]:
                    if self.slice_framepos == 0:
                        video_slice = raw_video_slice[:self.max_frames, ...]
                    elif self.slice_framepos == 1:
                        video_slice = raw_video_slice[-self.max_frames:, ...]
                    else:
                        sample_indx = np.linspace(0, raw_video_slice.shape[0] - 1, num=self.max_frames, dtype=int)
                        video_slice = raw_video_slice[sample_indx, ...]
                else:
                    video_slice = raw_video_slice

                video_slice = self.rawVideoExtractor.process_frame_order(video_slice, frame_order=self.frame_order)

                slice_len = video_slice.shape[0]
                max_video_length[i] = max_video_length[i] if max_video_length[i] > slice_len else slice_len
                if slice_len < 1:
                    pass
                else:
                    video[i][:slice_len, ...] = video_slice
            else:
                print("video path: {} error. video id: {}".format(video_path, video_id))

        for i, v_length in enumerate(max_video_length):
            video_mask[i][:v_length] = [1] * v_length

        return video, video_mask

    def _get_captions(self, titles):
        if isinstance(titles, str):
            titles = [titles]

        n_text = len(titles)

        pairs_text = np.zeros((n_text, self.max_words), dtype=np.long)
        pairs_mask = np.zeros((n_text, self.max_words), dtype=np.long)

        for idx in range(n_text):
            title = titles[idx]
            words = self.tokenizer.tokenize(title)

            # add begin
            words = [self.SPECIAL_TOKEN["CLS_TOKEN"]] + words
            total_length_with_CLS = self.max_words - 1
            if len(words) > total_length_with_CLS:
                words = words[:total_length_with_CLS]
            # add end
            words = words + [self.SPECIAL_TOKEN["SEP_TOKEN"]]

            input_ids = self.tokenizer.convert_tokens_to_ids(words)
            input_mask = [1] * len(input_ids)
            while len(input_ids) < self.max_words:
                input_ids.append(0)
                input_mask.append(0)
            assert len(input_ids) == self.max_words
            assert len(input_mask) == self.max_words

            pairs_text[idx] = np.array(input_ids)
            pairs_mask[idx] = np.array(input_mask)

        return pairs_text, pairs_mask
        
    def __getitem__(self, idx):
        video_id, caption = self.sentences_dict[idx]

        pairs_text, pairs_mask, pairs_segment, choice_video_ids = self._get_text(video_id, caption)

        video, video_mask = self._get_rawframes(choice_video_ids)

        if self.subset == 'test':
            caps_text, caps_mask = self._get_captions(self.generate_caps_dict[video_id])
        elif self.subset == 'train':
            caps_text, caps_mask = self._get_captions(self.generate_caps_dict[video_id]['captions'])

        return pairs_text, pairs_mask, pairs_segment, video, video_mask,caps_text, caps_mask




if __name__ == '__main__':
    data_path =  '/bpfs/v2_mnt/VIS/wuwenhao/VATEX'
    video_path = '/bpfs/v2_mnt/VIS/wuwenhao/VATEX/VATEX_Videos'
    frame_path = '/bpfs/v2_mnt/VIS/wuwenhao/VATEX/VATEX_Frames'
    import sys  
    sys.path.append('..')
    from modules.tokenization_clip import SimpleTokenizer as ClipTokenizer
    tokenizer = ClipTokenizer()

    max_words = 32
    feature_framerate = 1
    max_frames = 12
    train_frame_order = 0
    slice_framepos = 2

    dataset = VATEX_DataLoader(
        subset="train",
        data_path=data_path,
        features_path=frame_path,
        max_words=max_words,
        feature_framerate=feature_framerate,
        tokenizer=tokenizer,
        max_frames=max_frames,
        frame_order=train_frame_order,
        slice_framepos=slice_framepos,
    )

    from torch.utils.data import DataLoader
    dataloader_msvd = DataLoader(
        dataset,
        batch_size=1,
        num_workers=0,
        shuffle=False,
        drop_last=False,
    )

    for i, (pairs_text, pairs_mask, pairs_segment, video, video_mask) in enumerate(dataloader_msvd):
        print('='*80)
        print(i)
        print('[pairs_text]:', pairs_text.shape)
        print('[pairs_mask]:', pairs_mask.shape)
        print('[pairs_segment]:', pairs_segment.shape)
        print('[video]:', video.shape)
        print('[video_mask]:', video_mask.shape)
