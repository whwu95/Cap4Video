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
from transformers import BertTokenizer

class MSRVTT_DataLoader(Dataset):
    """MSRVTT dataset loader."""

    def __init__(
            self,
            csv_path,
            features_path,
            tokenizer,
            max_words=30,
            feature_framerate=1.0,
            max_frames=100,
            image_resolution=224,
            frame_order=0,
            slice_framepos=0,
            generate_text_backbone_clip_bert=0,
            original_text_backbone_clip_bert=0,
            num_class_names=12,
            generate_images=None,
            load_video=True
    ):

        # self.data = pd.read_csv(csv_path)
        self.data = json.load(open(csv_path, 'r'))  # vid: {gt:, titles:} # file:MSRVTT_JSFUSION_test_titles.json
        self.features_path = features_path
        self.feature_framerate = feature_framerate
        self.max_words = max_words
        self.max_frames = max_frames
        self.tokenizer = tokenizer
        # 0: ordinary order; 1: reverse order; 2: random order.
        self.frame_order = frame_order
        self.strategy = 1
        # print('Using uniform sampling without random offset for validation.')
        assert self.frame_order in [0, 1, 2]
        # 0: cut from head frames; 1: cut from tail frames; 2: extract frames uniformly.
        self.slice_framepos = slice_framepos
        assert self.slice_framepos in [0, 1, 2]

        self.rawVideoExtractor = RawVideoExtractor(framerate=feature_framerate, size=image_resolution)
        self.rawFramesExtractor = RawFramesExtractor(
            num_segments=max_frames, size=image_resolution, random_shift=True, strategy=self.strategy)

        self.SPECIAL_TOKEN = {"CLS_TOKEN": "<|startoftext|>", "SEP_TOKEN": "<|endoftext|>",
                              "MASK_TOKEN": "[MASK]", "UNK_TOKEN": "[UNK]", "PAD_TOKEN": "[PAD]"}
        self.BERT_SPECIAL_TOKEN = {"CLS_TOKEN": "[CLS]", "SEP_TOKEN": "[SEP]",
                                   "MASK_TOKEN": "[MASK]", "UNK_TOKEN": "[UNK]", "PAD_TOKEN": "[PAD]"}
        self.test_web_title_data = json.load(open('data/MSRVTT_test_website_titles.json', 'r'))
        self.load_video = load_video

    def __len__(self):
        return len(self.data)

    def _get_text(self, video_id, sentence):
        choice_video_ids = [video_id]
        n_caption = len(choice_video_ids)

        k = n_caption
        pairs_text = np.zeros((k, self.max_words), dtype=np.long)
        pairs_mask = np.zeros((k, self.max_words), dtype=np.long)
        pairs_segment = np.zeros((k, self.max_words), dtype=np.long)

        for i, video_id in enumerate(choice_video_ids):
            words = self.tokenizer.tokenize(sentence)

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

    def _get_rawvideo(self, choice_video_ids):
        print('???????' * 20)
        video_mask = np.zeros((len(choice_video_ids), self.max_frames), dtype=np.long)
        max_video_length = [0] * len(choice_video_ids)

        # Pair x L x T x 3 x H x W
        video = np.zeros((len(choice_video_ids), self.max_frames, 1, 3,
                          self.rawVideoExtractor.size, self.rawVideoExtractor.size), dtype=np.float)

        for i, video_id in enumerate(choice_video_ids):
            # Individual for YoucokII dataset, due to it video format
            video_path = os.path.join(self.features_path, "{}.mp4".format(video_id))
            if os.path.exists(video_path) is False:
                video_path = video_path.replace(".mp4", ".webm")

            raw_video_data = self.rawVideoExtractor.get_video_data(video_path)  # -------
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

    def _get_rawframes(self, choice_video_ids):
        video_mask = np.zeros((len(choice_video_ids), self.max_frames), dtype=np.long)
        max_video_length = [0] * len(choice_video_ids)

        # Pair x L x T x 3 x H x W
        video = np.zeros((len(choice_video_ids), self.max_frames, 1, 3,
                          self.rawFramesExtractor.size, self.rawFramesExtractor.size), dtype=np.float)

        for i, video_id in enumerate(choice_video_ids):
            # Individual for YoucokII dataset, due to it video format
            video_path = os.path.join(self.features_path, "{}".format(video_id))  # folder

            raw_video_data = self.rawFramesExtractor.get_video_data(video_path)
            raw_video_data = raw_video_data['video']
            if len(raw_video_data.shape) > 3:
                raw_video_data_clip = raw_video_data
                # L x T x 3 x H x W
                raw_video_slice = self.rawFramesExtractor.process_raw_data(raw_video_data_clip)
                if self.max_frames < raw_video_slice.shape[0]:
                    if self.slice_framepos == 0:  # cut from head
                        video_slice = raw_video_slice[:self.max_frames, ...]
                    elif self.slice_framepos == 1:  # cut from tail
                        video_slice = raw_video_slice[-self.max_frames:, ...]
                    else:  # extract uniformly
                        sample_index = np.linspace(0, raw_video_slice.shape[0] - 1, num=self.max_frames, dtype=int)
                        video_slice = raw_video_slice[sample_index, ...]
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

    def _get_title(self, titles):
        if titles is None:
            titles = ' '
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

    def _get_web_title(self, titles):
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
        video_id = list(self.data.keys())[idx]
        sentence = self.data[video_id]['gt']
        title = self.data[video_id]['titles']

        pairs_text, pairs_mask, pairs_segment, choice_video_ids = self._get_text(video_id, sentence)

        if self.load_video == True:
            video, video_mask = self._get_rawframes(choice_video_ids)
        else:
            video = 1
            video_mask = 1

        title_text, title_mask = self._get_title(title)

        return pairs_text, pairs_mask, pairs_segment, video, video_mask, title_text, title_mask

class MSRVTT_TrainDataLoader(Dataset):
    """MSRVTT train dataset loader."""

    def __init__(
            self,
            csv_path,
            json_path,
            features_path,
            tokenizer,
            max_words=30,
            feature_framerate=1.0,
            max_frames=100,
            unfold_sentences=False,
            image_resolution=224,
            frame_order=0,
            slice_framepos=0,
            strategy=1,
            generate_text_backbone_clip_bert=0,
            original_text_backbone_clip_bert=1,
            num_class_names=12,
            generate_images=None,
            load_video=True,
            caption_aug = True
    ):

        self.csv = pd.read_csv(csv_path)
        self.data = json.load(open(json_path, 'r'))
        self.features_path = features_path
        self.feature_framerate = feature_framerate
        self.max_words = max_words
        self.max_frames = max_frames
        self.tokenizer = tokenizer
        # 0: ordinary order; 1: reverse order; 2: random order.
        self.frame_order = frame_order
        assert self.frame_order in [0, 1, 2]
        # 0: cut from head frames; 1: cut from tail frames; 2: extract frames uniformly.
        self.slice_framepos = slice_framepos
        assert self.slice_framepos in [0, 1, 2]

        self.unfold_sentences = unfold_sentences
        self.sample_len = 0
        self.strategy = strategy
        self.caption_aug = caption_aug
        if self.caption_aug == True:
            title_list = []
            for k, v in self.data['title'].items():
                if isinstance(v,list):
                    for i in range(len(v)):
                        title_list.append({'video_id': k, 'caption': v[i]})
                else:
                    title_list.append({'video_id': k, 'caption': v})
            self.data['sentences'].extend(title_list)

        if self.unfold_sentences:
            train_video_ids = list(self.csv['video_id'].values)
            self.sentences_dict = {}
            for itm in self.data['sentences']:
                if itm['video_id'] in train_video_ids:
                    self.sentences_dict[len(self.sentences_dict)] = (
                    itm['video_id'], itm['caption'])
            self.sample_len = len(self.sentences_dict)

        else:
            num_sentences = 0
            self.sentences = defaultdict(list)
            s_video_id_set = set()
            for itm in self.data['sentences']:
                self.sentences[itm['video_id']].append(itm['caption'])
                num_sentences += 1
                s_video_id_set.add(itm['video_id'])

            # Use to find the clips in the same video
            self.parent_ids = {}
            self.children_video_ids = defaultdict(list)
            for itm in self.data['videos']:
                vid = itm["video_id"]
                url_posfix = itm["url"].split("?v=")[-1]
                self.parent_ids[vid] = url_posfix
                self.children_video_ids[url_posfix].append(vid)
            self.sample_len = len(self.csv)


        self.rawVideoExtractor = RawVideoExtractor(framerate=feature_framerate, size=image_resolution)
        self.rawFramesExtractor = RawFramesExtractor(
            num_segments=max_frames, size=image_resolution, random_shift=True, strategy=self.strategy)
        self.SPECIAL_TOKEN = {"CLS_TOKEN": "<|startoftext|>", "SEP_TOKEN": "<|endoftext|>",
                              "MASK_TOKEN": "[MASK]", "UNK_TOKEN": "[UNK]", "PAD_TOKEN": "[PAD]"}
        self.BERT_SPECIAL_TOKEN = {"CLS_TOKEN": "[CLS]", "SEP_TOKEN": "[SEP]",
                                   "MASK_TOKEN": "[MASK]", "UNK_TOKEN": "[UNK]", "PAD_TOKEN": "[PAD]"}
        self.load_video = load_video

    def __len__(self):
        return self.sample_len


    def _get_text(self, video_id, caption=None):
        k = 1
        choice_video_ids = [video_id]
        pairs_text = np.zeros((k, self.max_words), dtype=np.long)
        pairs_mask = np.zeros((k, self.max_words), dtype=np.long)
        pairs_segment = np.zeros((k, self.max_words), dtype=np.long)

        for i, video_id in enumerate(choice_video_ids):
            if caption is not None:
                words = self.tokenizer.tokenize(caption)
            else:
                words = self._get_single_text(video_id)

            # add begin
            words = [self.SPECIAL_TOKEN["CLS_TOKEN"]] + words
            total_length_with_CLS = self.max_words - 1
            if len(words) > total_length_with_CLS:
                words = words[:total_length_with_CLS]
            # add end
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

    def _get_single_text(self, video_id):

        rind = len(self.sentences['video_id']) // 2
        caption = self.sentences[video_id][rind]
        words = self.tokenizer.tokenize(caption)
        return words

    def _get_title(self, video_id):
        # get the video title vis video_id str
        title = self.data['title'][video_id]
        words = self.tokenizer.tokenize(title)

        pairs_text = np.zeros((1, self.max_words), dtype=np.long)
        pairs_mask = np.zeros((1, self.max_words), dtype=np.long)

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

        pairs_text[0] = np.array(input_ids)
        pairs_mask[0] = np.array(input_mask)

        return pairs_text, pairs_mask

    def _get_titles(self, video_id):
        # get the video title vis video_id str
        titles = self.data['titles'][video_id]
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

    def _get_web_title(self, video_id):

        title = self.web_title_data[video_id]
        if title is None:
            title = ' '
        titles = [title]
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

    def _get_rawvideo(self, choice_video_ids):
        print('&' * 30)
        video_mask = np.zeros((len(choice_video_ids), self.max_frames), dtype=np.long)
        max_video_length = [0] * len(choice_video_ids)  # [0]

        video = np.zeros((len(choice_video_ids), self.max_frames, 1, 3,
                          self.rawVideoExtractor.size, self.rawVideoExtractor.size), dtype=np.float)

        for i, video_id in enumerate(choice_video_ids):
            # Individual for YoucokII dataset, due to its video format
            video_path = os.path.join(self.features_path, "{}.mp4".format(video_id))
            if os.path.exists(video_path) is False:
                video_path = video_path.replace(".mp4", ".webm")

            raw_video_data = self.rawVideoExtractor.get_video_data(video_path)
            raw_video_data = raw_video_data['video']
            if len(raw_video_data.shape) > 3:
                raw_video_data_clip = raw_video_data
                raw_video_slice = self.rawVideoExtractor.process_raw_data(
                    raw_video_data_clip)
                if self.max_frames < raw_video_slice.shape[0]:
                    if self.slice_framepos == 0:  # cut from head
                        video_slice = raw_video_slice[:self.max_frames, ...]
                    elif self.slice_framepos == 1:  # cut from tail
                        video_slice = raw_video_slice[-self.max_frames:, ...]
                    else:  # extract uniformly
                        sample_indx = np.linspace(0, raw_video_slice.shape[0] - 1, num=self.max_frames, dtype=int)
                        video_slice = raw_video_slice[sample_indx, ...]
                else:
                    video_slice = raw_video_slice
                video_slice = self.rawVideoExtractor.process_frame_order(video_slice,
                                                                         frame_order=self.frame_order)

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

    def _get_rawframes(self, choice_video_ids):

        video_mask = np.zeros((len(choice_video_ids), self.max_frames), dtype=np.long)
        max_video_length = [0] * len(choice_video_ids)

        # Pair x L x T x 3 x H x W
        video = np.zeros((len(choice_video_ids), self.max_frames, 1, 3,
                          self.rawFramesExtractor.size, self.rawFramesExtractor.size), dtype=np.float)

        for i, video_id in enumerate(choice_video_ids):
            # Individual for YoucokII dataset, due to it video format
            video_path = os.path.join(self.features_path, "{}".format(video_id))  # folder

            raw_video_data = self.rawFramesExtractor.get_video_data(video_path)
            raw_video_data = raw_video_data['video']
            if len(raw_video_data.shape) > 3:
                raw_video_data_clip = raw_video_data
                # L x T x 3 x H x W
                raw_video_slice = self.rawFramesExtractor.process_raw_data(raw_video_data_clip)
                if self.max_frames < raw_video_slice.shape[0]:
                    if self.slice_framepos == 0:  # cut from head
                        video_slice = raw_video_slice[:self.max_frames, ...]
                    elif self.slice_framepos == 1:  # cut from tail
                        video_slice = raw_video_slice[-self.max_frames:, ...]
                    else:  # extract uniformly
                        sample_index = np.linspace(0, raw_video_slice.shape[0] - 1, num=self.max_frames, dtype=int)
                        video_slice = raw_video_slice[sample_index, ...]
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

    def __getitem__(self, idx):
        if self.unfold_sentences:
            video_id, caption = self.sentences_dict[idx]
        else:
            video_id, caption = self.csv['video_id'].values[idx], None
        pairs_text, pairs_mask, pairs_segment, choice_video_ids = self._get_text(video_id, caption)

        if self.load_video == True:
            video, video_mask = self._get_rawframes(choice_video_ids)
        else:
            video = 1
            video_mask = 1

        title_text, title_mask = self._get_titles(video_id)

        return pairs_text, pairs_mask, pairs_segment, video, video_mask, title_text, title_mask
