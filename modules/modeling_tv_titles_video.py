from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

import torch
from torch import nn
import torch.nn.functional as F

from modules.until_module import PreTrainedModel, AllGather, CrossEn, cosface
from modules.module_cross import CrossModel, CrossConfig, Transformer as TransformerClip

from modules.module_clip import CLIP, convert_weights
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

from modules.co_attention_transformer_module import Co_attention_block

logger = logging.getLogger(__name__)
allgather = AllGather.apply

class CLIP4ClipPreTrainedModel(PreTrainedModel, nn.Module):
    def __init__(self, cross_config, *inputs, **kwargs):
        super(CLIP4ClipPreTrainedModel, self).__init__(cross_config)
        self.cross_config = cross_config
        self.clip = None
        self.cross = None

    @classmethod
    def from_pretrained(cls, cross_model_name, state_dict=None, cache_dir=None, type_vocab_size=2, *inputs, **kwargs):

        task_config = None
        if "task_config" in kwargs.keys():
            task_config = kwargs["task_config"]
            if not hasattr(task_config, "local_rank"):
                task_config.__dict__["local_rank"] = 0
            elif task_config.local_rank == -1:
                task_config.local_rank = 0

        if state_dict is None: state_dict = {}
        if hasattr(task_config, 'pretrained_clip_name'):
            pretrained_clip_name = task_config.pretrained_clip_name
        clip_state_dict = CLIP.get_config(pretrained_clip_name=pretrained_clip_name)
        for key, val in clip_state_dict.items():
            new_key = "clip." + key
            if new_key not in state_dict:
                state_dict[new_key] = val.clone()
        cross_config, _ = CrossConfig.get_config(cross_model_name, cache_dir, type_vocab_size, state_dict=None, task_config=task_config)
        model = cls(cross_config, clip_state_dict, *inputs, **kwargs) # -----------

        if model.linear_patch == "3d":
            contain_conv2 = False
            for key in state_dict.keys():
                if key.find("visual.conv2.weight") > -1:
                    contain_conv2 = True
                    break
            if contain_conv2 is False and hasattr(model.clip.visual, "conv2"):
                cp_weight = state_dict["clip.visual.conv1.weight"].clone()
                kernel_size = model.clip.visual.conv2.weight.size(2)
                conv2_size = model.clip.visual.conv2.weight.size()
                conv2_size = list(conv2_size)

                left_conv2_size = conv2_size.copy()
                right_conv2_size = conv2_size.copy()
                left_conv2_size[2] = (kernel_size - 1) // 2
                right_conv2_size[2] = kernel_size - 1 - left_conv2_size[2]

                left_zeros, right_zeros = None, None
                if left_conv2_size[2] > 0:
                    left_zeros = torch.zeros(*tuple(left_conv2_size), dtype=cp_weight.dtype, device=cp_weight.device)
                if right_conv2_size[2] > 0:
                    right_zeros = torch.zeros(*tuple(right_conv2_size), dtype=cp_weight.dtype, device=cp_weight.device)

                cat_list = []
                if left_zeros != None: cat_list.append(left_zeros)
                cat_list.append(cp_weight.unsqueeze(2))
                if right_zeros != None: cat_list.append(right_zeros)
                cp_weight = torch.cat(cat_list, dim=2)

                state_dict["clip.visual.conv2.weight"] = cp_weight

        if model.sim_header == 'tightTransf':
            contain_cross = False
            for key in state_dict.keys():
                if key.find("cross.transformer") > -1:
                    contain_cross = True
                    break
            if contain_cross is False:
                for key, val in clip_state_dict.items():
                    if key == "positional_embedding":
                        state_dict["cross.embeddings.position_embeddings.weight"] = val.clone()
                        continue
                    if key.find("transformer.resblocks") == 0:
                        num_layer = int(key.split(".")[2])

                        # cut from beginning
                        if num_layer < task_config.cross_num_hidden_layers:
                            state_dict["cross."+key] = val.clone()
                            continue

        if model.sim_header in ["seqLSTM", "seqTransf"]:
            contain_frame_position = False
            for key in state_dict.keys():
                if key.find("frame_position_embeddings") > -1:
                    contain_frame_position = True
                    break
            if contain_frame_position is False:
                for key, val in clip_state_dict.items():
                    if key == "positional_embedding":
                        state_dict["frame_position_embeddings.weight"] = val.clone()
                        continue
                    if "seqTransf" in model.sim_header and key.find("transformer.resblocks") == 0:
                        num_layer = int(key.split(".")[2])
                        # cut from beginning
                        if num_layer < task_config.cross_num_hidden_layers:
                            state_dict[key.replace("transformer.", "transformerClip.")] = val.clone()
                            continue
        ## <=== End of initialization trick

        if state_dict is not None:
            model = cls.init_preweight(model, state_dict, task_config=task_config)

        return model

def show_log(task_config, info):
    if task_config is None or task_config.local_rank == 0:
        logger.warning(info)

def update_attr(target_name, target_config, target_attr_name, source_config, source_attr_name, default_value=None):
    if hasattr(source_config, source_attr_name):
        if default_value is None or getattr(source_config, source_attr_name) != default_value:
            setattr(target_config, target_attr_name, getattr(source_config, source_attr_name))
            show_log(source_config, "Set {}.{}: {}.".format(target_name,
                                                            target_attr_name, getattr(target_config, target_attr_name)))
    return target_config

def check_attr(target_name, task_config):
    return hasattr(task_config, target_name) and task_config.__dict__[target_name]


class CLIP4Clip(CLIP4ClipPreTrainedModel):
    def __init__(self, cross_config, clip_state_dict, task_config):
        super(CLIP4Clip, self).__init__(cross_config)
        self.task_config = task_config
        self.ignore_video_index = -1

        assert self.task_config.max_words + self.task_config.max_frames <= cross_config.max_position_embeddings

        self._stage_one = True
        self._stage_two = False

        show_log(task_config, "Stage-One:{}, Stage-Two:{}".format(self._stage_one, self._stage_two))

        self.loose_type = False
        if self._stage_one and check_attr('loose_type', self.task_config):
            self.loose_type = True
            show_log(task_config, "Test retrieval by loose type.")

        # CLIP Encoders: From OpenAI: CLIP [https://github.com/openai/CLIP] ===>
        vit = "visual.proj" in clip_state_dict
        assert vit
        if vit:
            vision_width = clip_state_dict["visual.conv1.weight"].shape[0]
            vision_layers = len([k for k in clip_state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
            vision_patch_size = clip_state_dict["visual.conv1.weight"].shape[-1]
            grid_size = round((clip_state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
            image_resolution = vision_patch_size * grid_size
        else:
            counts: list = [len(set(k.split(".")[2] for k in clip_state_dict if k.startswith(f"visual.layer{b}"))) for b in
                            [1, 2, 3, 4]]
            vision_layers = tuple(counts)
            vision_width = clip_state_dict["visual.layer1.0.conv1.weight"].shape[0]
            output_width = round((clip_state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
            vision_patch_size = None
            assert output_width ** 2 + 1 == clip_state_dict["visual.attnpool.positional_embedding"].shape[0]
            image_resolution = output_width * 32

        embed_dim = clip_state_dict["text_projection"].shape[1]
        context_length = clip_state_dict["positional_embedding"].shape[0]
        vocab_size = clip_state_dict["token_embedding.weight"].shape[0]
        transformer_width = clip_state_dict["ln_final.weight"].shape[0]
        transformer_heads = transformer_width // 64
        transformer_layers = len(set(k.split(".")[2] for k in clip_state_dict if k.startswith(f"transformer.resblocks")))

        show_log(task_config, "\t embed_dim: {}".format(embed_dim))
        show_log(task_config, "\t image_resolution: {}".format(image_resolution))
        show_log(task_config, "\t vision_layers: {}".format(vision_layers))
        show_log(task_config, "\t vision_width: {}".format(vision_width))
        show_log(task_config, "\t vision_patch_size: {}".format(vision_patch_size))
        show_log(task_config, "\t context_length: {}".format(context_length))
        show_log(task_config, "\t vocab_size: {}".format(vocab_size))
        show_log(task_config, "\t transformer_width: {}".format(transformer_width))
        show_log(task_config, "\t transformer_heads: {}".format(transformer_heads))
        show_log(task_config, "\t transformer_layers: {}".format(transformer_layers))

        self.linear_patch = '2d'
        if hasattr(task_config, "linear_patch"):
            self.linear_patch = task_config.linear_patch
            show_log(task_config, "\t\t linear_patch: {}".format(self.linear_patch))

        # use .float() to avoid overflow/underflow from fp16 weight. https://github.com/openai/CLIP/issues/40
        cut_top_layer = 0
        show_log(task_config, "\t cut_top_layer: {}".format(cut_top_layer))
        self.clip = CLIP(
            embed_dim,
            image_resolution, vision_layers-cut_top_layer, vision_width, vision_patch_size,
            context_length, vocab_size, transformer_width, transformer_heads, transformer_layers-cut_top_layer,
            linear_patch=self.linear_patch
        ).float()

        self.co_connetion_transformer_model_block = nn.Sequential(*[Co_attention_block(hidden_size=embed_dim, num_attention_heads=transformer_heads, dropout_rate=0.1) for i in range(1)])

        for key in ["input_resolution", "context_length", "vocab_size"]:
            if key in clip_state_dict:
                del clip_state_dict[key]

        convert_weights(self.clip)
        # <=== End of CLIP Encoders

        self.sim_header = 'meanP'
        if hasattr(task_config, "sim_header"):
            self.sim_header = task_config.sim_header
            show_log(task_config, "\t sim_header: {}".format(self.sim_header))
        if self.sim_header == "tightTransf": assert self.loose_type is False

        self.interaction = 'dp'
        if hasattr(task_config, "interaction"):
            self.interaction = task_config.interaction
            show_log(task_config, "\t interaction: {}".format(self.interaction))

        cross_config.max_position_embeddings = context_length
        if self.loose_type is False:
            # Cross Encoder ===>
            cross_config = update_attr("cross_config", cross_config, "num_hidden_layers", self.task_config, "cross_num_hidden_layers")
            self.cross = CrossModel(cross_config)
            # <=== End of Cross Encoder
            self.similarity_dense = nn.Linear(cross_config.hidden_size, 1)

        if self.sim_header in ["seqLSTM", "seqTransf"]:
            self.frame_position_embeddings = nn.Embedding(cross_config.max_position_embeddings, cross_config.hidden_size)
        if "seqTransf" in self.sim_header:
            self.transformerClip = TransformerClip(width=transformer_width, layers=self.task_config.cross_num_hidden_layers, heads=transformer_heads,)

        if self.sim_header == "seqLSTM":
            self.lstm_visual = nn.LSTM(input_size=cross_config.hidden_size, hidden_size=cross_config.hidden_size,
                                       batch_first=True, bidirectional=False, num_layers=1)

        self.fuse_weight_fc = nn.Linear(transformer_width, 2)

        self.query_weight_fc = nn.Sequential(
                    nn.Linear(transformer_width, transformer_width), nn.ReLU(inplace=True),
                    nn.Linear(transformer_width, 1))

        self.text_pool_type = task_config.text_pool_type
        if self.text_pool_type in ['weight_l', 'weight_g']:
            self.title_weight_fc = nn.Sequential(
                        nn.Linear(transformer_width, transformer_width), nn.ReLU(inplace=True),
                        nn.Linear(transformer_width, 1))
        self.k = task_config.k

        if self.interaction == 'wti':
            if self.task_config.wti_arch == 1:
                self.text_weight_fc = nn.Linear(transformer_width, 1)
                self.video_weight_fc = nn.Linear(transformer_width, 1)
            elif self.task_config.wti_arch == 2:
                self.text_weight_fc = nn.Sequential(
                    nn.Linear(transformer_width, transformer_width), nn.ReLU(inplace=True),
                    nn.Linear(transformer_width, 1))
                self.video_weight_fc = nn.Sequential(
                    nn.Linear(transformer_width, transformer_width), nn.ReLU(inplace=True),
                    nn.Linear(transformer_width, 1))
            elif self.task_config.wti_arch == 3:
                self.text_weight_fc = nn.Sequential(
                    nn.Linear(transformer_width, transformer_width), nn.ReLU(inplace=True),
                    nn.Linear(transformer_width, transformer_width), nn.ReLU(inplace=True),
                    nn.Linear(transformer_width, 1))
                self.video_weight_fc = nn.Sequential(
                    nn.Linear(transformer_width, transformer_width), nn.ReLU(inplace=True),
                    nn.Linear(transformer_width, transformer_width), nn.ReLU(inplace=True),
                    nn.Linear(transformer_width, 1))

        if self.text_pool_type in ['transf_avg']:
            self.num_captions = 30
            self.sentence_position_embeddings = nn.Embedding(self.num_captions, embed_dim)  # 时序transformer的位置编码
            self.caption_transformer_layer = nn.TransformerEncoderLayer(d_model=transformer_width,
                                                                        nhead=transformer_heads,
                                                                        dim_feedforward=transformer_width, dropout=0,
                                                                        batch_first=True)
            self.caption_transformer_encoder = nn.TransformerEncoder(self.caption_transformer_layer, num_layers=2)
            self.text_position_embeddings = nn.Embedding(context_length, embed_dim)

        self.loss_fct = CrossEn()
        self.apply(self.init_weights)

    def forward(self, text_ids, attention_mask, video, video_mask, title_ids, title_mask, train_video):

        text_emb, video_emb, title_emb = self.get_text_video_title_output(text_ids, video, title_ids, title_mask, attention_mask, train_video)
        if self.training:
            loss = 0.
            if train_video == True:
                sim_matrix = self.get_video_text_similarity_logits(text_emb, video_emb, title_emb, attention_mask, video_mask, title_mask,loose_type=self.loose_type)
            else:
                sim_matrix = self.get_titles_similarity_logits(text_emb, video_emb, title_emb, attention_mask, video_mask, title_mask, loose_type=self.loose_type)

            sim_loss1 = self.loss_fct(sim_matrix)
            sim_loss2 = self.loss_fct(sim_matrix.T)
            sim_loss = (sim_loss1 + sim_loss2) / 2
            loss = sim_loss
            return loss
        else:
            if train_video == True:
                sim_matrix = self.get_video_text_similarity_logits(text_emb, video_emb, title_emb, attention_mask,video_mask, title_mask, loose_type=self.loose_type)
            else:
                sim_matrix = self.get_titles_similarity_logits(text_emb, video_emb, title_emb, attention_mask,video_mask, title_mask, loose_type=self.loose_type)

            return None

    def get_text_output(self, input_ids):
        bs_pair = input_ids.size(0)
        n_text = input_ids.size(1)
        input_ids = input_ids.view(-1, input_ids.shape[-1])
        sequence_hidden = self.clip.encode_text(input_ids, return_hidden=True)[1].float()
        sequence_hidden = sequence_hidden.view(bs_pair, n_text, -1, sequence_hidden.size(-1))
        
        if n_text == 1:
            sequence_hidden = sequence_hidden.view(bs_pair, -1, sequence_hidden.size(-1))
        return sequence_hidden

    def get_video_output(self, video):
        # video: b, 1, t, 1, c, h, w
        b, pair, ts, bs, channel, h, w = video.shape
        # [b, 1, t, 1, c, h, w] -> [b*t, c, h, w]
        video = video.view(b * pair * bs * ts, channel, h, w)
        video_frame = bs * ts
        
        # [bs * t, c, h, w] -> [bs * t, dim]
        visual_hidden = self.clip.encode_image(video, video_frame=video_frame).float()
        # [bs * t, dim] -> [bs, t, dim]
        visual_hidden = visual_hidden.view(-1, video_frame, visual_hidden.size(-1))
        return visual_hidden

    def get_text_video_output(self, input_ids, video):
        sequence_output = self.get_text_output(input_ids)
        visual_output = self.get_video_output(video)
        return sequence_output, visual_output

    def get_text_video_title_output(self, text_ids, video, title_ids, title_mask=None, text_mask=None, train_video = True):

        sequence_output = self.get_text_output(text_ids)
        if train_video == True:
            visual_output = self.get_video_output(video)
        else:
            visual_output = torch.tensor([1], device=video.device)
        title_emb = self.get_text_output(title_ids)
        return sequence_output, visual_output, title_emb

    def _get_cross_output(self, sequence_output, visual_output, attention_mask, video_mask):
        # haven't uesd, TODO
        concat_features = torch.cat((sequence_output, visual_output), dim=1)  # concatnate tokens and frames
        concat_mask = torch.cat((attention_mask, video_mask), dim=1)
        text_type_ = torch.zeros_like(attention_mask)
        video_type_ = torch.ones_like(video_mask)
        concat_type = torch.cat((text_type_, video_type_), dim=1)
        cross_layers, pooled_output = self.cross(concat_features, concat_type, concat_mask, output_all_encoded_layers=True)
        cross_output = cross_layers[-1]
        return cross_output, pooled_output, concat_mask

    def get_text_sep_feat(self, text_feat, text_mask):
        # text_mask: [bs_text, max_words] or [bs_text, n_text, max_words]
        # text_feat: [bs_text, n_words, dim] or [bs_text, n_text, n_words, dim]
        # output: [bs_text, n_text, dim]
        n_dim = text_feat.dim()
        text_feat = text_feat.contiguous()
        if n_dim == 3: # n_dim=3表示文本句子描述
            text_feat = text_feat[torch.arange(text_feat.shape[0]), torch.sum(text_mask, dim=-1) - 1, :]
            text_feat = text_feat.unsqueeze(1).contiguous()
        elif n_dim == 4:
            bs_pair, n_text, n_word, text_dim = text_feat.shape
            text_feat = text_feat.view(bs_pair * n_text, n_word, text_dim)
            text_mask = text_mask.view(bs_pair * n_text, n_word)
            text_feat = text_feat[torch.arange(text_feat.shape[0]), torch.sum(text_mask, dim=-1) - 1, :]
            text_feat = text_feat.view(bs_pair, n_text, text_dim)
        return text_feat

    def _mean_pooling_for_similarity_text(self, sequence_output, attention_mask):
        attention_mask_un = attention_mask.to(dtype=torch.float).unsqueeze(-1)
        attention_mask_un[:, 0, :] = 0.
        sequence_output = sequence_output * attention_mask_un
        text_out = torch.sum(sequence_output, dim=1) / torch.sum(attention_mask_un, dim=1, dtype=torch.float)
        return text_out

    def _mean_pooling_for_similarity_video(self, visual_output, video_mask,):
        video_mask_un = video_mask.to(dtype=torch.float).unsqueeze(-1)
        visual_output = visual_output * video_mask_un
        video_mask_un_sum = torch.sum(video_mask_un, dim=1, dtype=torch.float)
        video_mask_un_sum[video_mask_un_sum == 0.] = 1.
        video_out = torch.sum(visual_output, dim=1) / video_mask_un_sum
        return video_out

    def _mean_pooling_for_similarity(self, sequence_output, visual_output, attention_mask, video_mask,):
        text_out = self._mean_pooling_for_similarity_text(sequence_output, attention_mask)
        video_out = self._mean_pooling_for_similarity_video(visual_output, video_mask)

        return text_out, video_out

    def agg_video_feat(self, visual_output, video_mask, sim_header="meanP"):
        visual_output = visual_output.contiguous()
        if sim_header == "meanP":
            # Default: Parameter-free type
            pass
        elif sim_header == "seqLSTM":
            # Sequential type: LSTM
            visual_output_original = visual_output
            visual_output = pack_padded_sequence(visual_output, torch.sum(video_mask, dim=-1).cpu(),
                                                 batch_first=True, enforce_sorted=False)
            visual_output, _ = self.lstm_visual(visual_output)
            if self.training: self.lstm_visual.flatten_parameters()
            visual_output, _ = pad_packed_sequence(visual_output, batch_first=True)
            visual_output = torch.cat((visual_output, visual_output_original[:, visual_output.size(1):, ...].contiguous()), dim=1)
            visual_output = visual_output + visual_output_original
        elif "seqTransf" in sim_header:
            # Sequential type: Transformer Encoder
            visual_output_original = visual_output
            seq_length = visual_output.size(1)
            position_ids = torch.arange(seq_length, dtype=torch.long, device=visual_output.device)
            position_ids = position_ids.unsqueeze(0).expand(visual_output.size(0), -1)
            frame_position_embeddings = self.frame_position_embeddings(position_ids)
            visual_output = visual_output + frame_position_embeddings

            extended_video_mask = (1.0 - video_mask.unsqueeze(1)) * -1000000.0
            extended_video_mask = extended_video_mask.expand(-1, video_mask.size(1), -1)
            visual_output = visual_output.permute(1, 0, 2)  # NLD -> LND
            visual_output = self.transformerClip(visual_output, extended_video_mask)
            visual_output = visual_output.permute(1, 0, 2)  # LND -> NLD
            visual_output = visual_output + visual_output_original
        return visual_output

    def dot_product_logits(self, sequence_output, visual_output, text_mask, video_mask):
        sequence_output = self.get_text_sep_feat(sequence_output, text_mask)  # B x 1 x D
        if self.training:
            visual_output = allgather(visual_output, self.task_config)
            video_mask = allgather(video_mask, self.task_config)
            sequence_output = allgather(sequence_output, self.task_config)
            torch.distributed.barrier()  

        sequence_output = sequence_output.squeeze(1)  # B x 1 x D -> B x D
        sequence_output = sequence_output / sequence_output.norm(dim=-1, keepdim=True)
        visual_output = visual_output / visual_output.norm(dim=-1, keepdim=True)
        visual_output = self._mean_pooling_for_similarity_video(visual_output, video_mask)      # B x N_v x D -> B x D
        visual_output = visual_output / visual_output.norm(dim=-1, keepdim=True)

        retrieve_logits = torch.matmul(sequence_output, visual_output.t())  # n_t n_v  
        if self.training:
            logit_scale = self.clip.logit_scale.exp()
            retrieve_logits = logit_scale * retrieve_logits
            return retrieve_logits
        else:
            return retrieve_logits

    def wti_interaction(self, text_feat, video_feat, text_mask, video_mask):
        if self.training and torch.cuda.is_available():  # batch merge here
            text_feat = allgather(text_feat, self.task_config)
            video_feat = allgather(video_feat, self.task_config)
            text_mask = allgather(text_mask, self.task_config)
            video_mask = allgather(video_mask, self.task_config)
            torch.distributed.barrier()  # force sync

        if self.interaction == 'wti':
            text_weight = self.text_weight_fc(text_feat).squeeze(2)  # B x N_t x D -> B x N_t
            text_weight.masked_fill_(torch.tensor((1 - text_mask), dtype=torch.bool), float("-inf"))
            text_weight = torch.softmax(text_weight, dim=-1)  # B x N_t

            video_weight = self.video_weight_fc(video_feat).squeeze(2) # B x N_v x D -> B x N_v
            video_weight.masked_fill_((1 - video_mask).clone().detach().bool(), float("-inf"))
            video_weight = torch.softmax(video_weight, dim=-1)  # B x N_v

        text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
        video_feat = video_feat / video_feat.norm(dim=-1, keepdim=True)

        retrieve_logits = torch.einsum('atd,bvd->abtv', [text_feat, video_feat])
        retrieve_logits = torch.einsum('abtv,at->abtv', [retrieve_logits, text_mask])
        retrieve_logits = torch.einsum('abtv,bv->abtv', [retrieve_logits, video_mask])
        text_sum = text_mask.sum(-1)
        video_sum = video_mask.sum(-1)

        # max for video token
        if self.interaction == 'ti':  # token-wise interaction
            t2v_logits, max_idx1 = retrieve_logits.max(dim=-1)  # abtv -> abt
            v2t_logits, max_idx2 = retrieve_logits.max(dim=-2)  # abtv -> abv
            t2v_logits = torch.sum(t2v_logits, dim=2) / (text_sum.unsqueeze(1))
            v2t_logits = torch.sum(v2t_logits, dim=2) / (video_sum.unsqueeze(0))
            retrieve_logits = (t2v_logits + v2t_logits) / 2.0

        elif self.interaction == 'wti':  # weighted token-wise interaction
            t2v_logits, max_idx1 = retrieve_logits.max(dim=-1)  # abtv -> abt
            t2v_logits = torch.einsum('abt,at->ab', [t2v_logits, text_weight])

            v2t_logits, max_idx2 = retrieve_logits.max(dim=-2)  # abtv -> abv
            v2t_logits = torch.einsum('abv,bv->ab', [v2t_logits, video_weight])
            retrieve_logits = (t2v_logits + v2t_logits) / 2.0

        if self.training:
            logit_scale = self.clip.logit_scale.exp()
            retrieve_logits = logit_scale * retrieve_logits
            return retrieve_logits
        else:
            return retrieve_logits

    def _loose_similarity(self, sequence_output, visual_output, attention_mask, video_mask, sim_header="meanP"):
        sequence_output = sequence_output.contiguous()

        if self.interaction == 'dp':
            retrieve_logits = self.dot_product_logits(sequence_output, visual_output, attention_mask, video_mask)
        elif self.interaction in ['ti', 'wti']:
            retrieve_logits = self.wti_interaction(sequence_output, visual_output, attention_mask, video_mask)
        else:
            raise NotImplementedError
        return retrieve_logits

    def _cross_similarity(self, sequence_output, visual_output, attention_mask, video_mask):
        sequence_output, visual_output = sequence_output.contiguous(), visual_output.contiguous()

        b_text, s_text, h_text = sequence_output.size()
        b_visual, s_visual, h_visual = visual_output.size()

        retrieve_logits_list = []

        step_size = b_text      # set smaller to reduce memory cost
        split_size = [step_size] * (b_text // step_size)
        release_size = b_text - sum(split_size)
        if release_size > 0:
            split_size += [release_size]

        # due to clip text branch return the last hidden
        attention_mask = torch.ones(sequence_output.size(0), 1)\
            .to(device=attention_mask.device, dtype=attention_mask.dtype)

        sequence_output_splits = torch.split(sequence_output, split_size, dim=0)
        attention_mask_splits = torch.split(attention_mask, split_size, dim=0)
        for i in range(len(split_size)):
            sequence_output_row = sequence_output_splits[i]
            attention_mask_row = attention_mask_splits[i]
            sequence_output_l = sequence_output_row.unsqueeze(1).repeat(1, b_visual, 1, 1)
            sequence_output_l = sequence_output_l.view(-1, s_text, h_text)
            attention_mask_l = attention_mask_row.unsqueeze(1).repeat(1, b_visual, 1)
            attention_mask_l = attention_mask_l.view(-1, s_text)

            step_truth = sequence_output_row.size(0)
            visual_output_r = visual_output.unsqueeze(0).repeat(step_truth, 1, 1, 1)
            visual_output_r = visual_output_r.view(-1, s_visual, h_visual)
            video_mask_r = video_mask.unsqueeze(0).repeat(step_truth, 1, 1)
            video_mask_r = video_mask_r.view(-1, s_visual)

            cross_output, pooled_output, concat_mask = \
                self._get_cross_output(sequence_output_l, visual_output_r, attention_mask_l, video_mask_r)
            retrieve_logits_row = self.similarity_dense(pooled_output).squeeze(-1).view(step_truth, b_visual)

            retrieve_logits_list.append(retrieve_logits_row)

        retrieve_logits = torch.cat(retrieve_logits_list, dim=0)
        return retrieve_logits

    def get_similarity_logits(self, sequence_output, visual_output, attention_mask, video_mask, loose_type=False):
        if loose_type:
            assert self.sim_header in ["meanP", "seqLSTM", "seqTransf"]
            retrieve_logits = self._loose_similarity(sequence_output, visual_output, attention_mask, video_mask, sim_header=self.sim_header)
            return retrieve_logits
        else:
            assert self.sim_header in ["tightTransf"]
            retrieve_logits = self._cross_similarity(sequence_output, visual_output, attention_mask, video_mask, )

            return retrieve_logits


    def get_video_text_similarity_logits(self, text_feat, video_feat, title_feat, text_mask, video_mask, title_mask, loose_type=False):
        # [bs_text, 1, max_words] -> [bs_text, max_words]
        text_mask = text_mask.view(-1, text_mask.shape[-1])
        # [bs_video, 1, max_words] -> [bs_video, max_words]
        video_mask = video_mask.view(-1, video_mask.shape[-1])

        title_output = self.get_text_sep_feat(title_feat,title_mask)

        cross_video_mask = video_mask.reshape(video_mask.shape[0],1,1,video_mask.shape[-1])
        cross_titles_mask = torch.ones((title_mask.shape[0],title_mask.shape[1]),device=title_output.device)
        cross_titles_mask = cross_titles_mask.reshape(cross_titles_mask.shape[0],1,1,cross_titles_mask.shape[-1])

        for co_layer in self.co_connetion_transformer_model_block:
            video_feat, title_output, co_attention_probs = co_layer(video_feat,cross_video_mask,title_output, cross_titles_mask)

        visual_output = self.agg_video_feat(video_feat, video_mask, self.sim_header) ##经过seq transfomer
        qv_logits = self.get_similarity_logits(text_feat, visual_output, text_mask, video_mask, loose_type)
        retrieve_logits = qv_logits
        return retrieve_logits

    def get_text_title_similarity_logits(self, text_output, title_output, text_mask, title_mask):
        # dp
        title_mask = title_mask.view(-1, title_mask.shape[-1])
        text_output = self.get_text_sep_feat(text_output, text_mask).squeeze(1)  # B x 1 x D -> B x D
        title_output = self.get_text_sep_feat(title_output, title_mask).squeeze(1)  # B x 1 x D -> B x D
        if self.training:
            text_output = allgather(text_output, self.task_config)
            title_output = allgather(title_output, self.task_config)
            torch.distributed.barrier()  

        text_output = text_output / text_output.norm(dim=-1, keepdim=True)
        title_output = title_output / title_output.norm(dim=-1, keepdim=True)

        retrieve_logits = torch.matmul(text_output, title_output.t())  # n_title n_cap
        if self.training:
            logit_scale = self.clip.logit_scale.exp()
            retrieve_logits = logit_scale * retrieve_logits
            return retrieve_logits
        else:
            return retrieve_logits

    def get_titles_similarity_logits(self, text_feat, video_feat, title_feat, text_mask, video_mask, title_mask,loose_type=False):
        text_mask = text_mask.view(-1, text_mask.shape[-1])
        video_mask = video_mask.view(-1, video_mask.shape[-1])
        qt_logits = self.get_text_titles_similarity_logits(text_feat, title_feat, text_mask, title_mask, video_feat,video_mask)
        retrieve_logits = qt_logits
        return retrieve_logits

    def get_text_titles_similarity_logits(self, text_output, title_output, text_mask, title_mask, video_feat,video_mask):

        title_output = self.get_text_sep_feat(title_output, title_mask)
        if self.text_pool_type in ['transf_avg']:
            # b t c
            x_original = title_output
            seq_length = title_output.shape[1]
            position_ids = torch.arange(seq_length, dtype=torch.long, device=title_output.device)
            position_ids = position_ids.unsqueeze(0).expand(title_output.size(0), -1)
            sentence_position_embeddings = self.sentence_position_embeddings(position_ids)
            title_output = title_output + sentence_position_embeddings
            title_output = self.caption_transformer_encoder(title_output)

        if self.training:
            text_output = allgather(text_output, self.task_config)
            title_output = allgather(title_output, self.task_config)
            text_mask = allgather(text_mask, self.task_config)
            torch.distributed.barrier()

        title_ori = title_output
        text_embed = self.get_text_sep_feat(text_output, text_mask).squeeze(1)
        ################# title pooling begin ##############
        if self.text_pool_type == 'clip_top1':
            title_embed_pooled = title_output[:, 0]

        elif self.text_pool_type in ['avg', 'transf_avg']:
            title_embed_pooled = title_output.mean(dim=1)

        elif self.text_pool_type == 'topk':
            bs_text, embed_dim = text_embed.shape
            sims = title_output @ text_embed.t()
            sims_topk = torch.topk(sims, self.k, dim=1)[1]
            title_output = title_output.unsqueeze(-1).expand(-1, -1, -1, bs_text)
            sims_topk = sims_topk.unsqueeze(2).expand(-1, -1, embed_dim, -1)
            title_embeds_topk = torch.gather(title_output, dim=1, index=sims_topk)
            title_embed_pooled = title_embeds_topk.sum(dim=1)
            title_embed_pooled = title_embed_pooled.permute(0, 2, 1)

        ################# title pooling end ##############

        text_embed = text_embed / text_embed.norm(dim=-1, keepdim=True)
        title_embed_pooled = title_embed_pooled / title_embed_pooled.norm(dim=-1, keepdim=True)

        if self.text_pool_type in ['clip_top1', 'avg', 'transf_avg']:
            # bs_text x bs_title
            q2t_logits = torch.mm(text_embed, title_embed_pooled.t())

        if self.text_pool_type in ['clip_top1', 'avg', 'transf_avg']:
            retrieve_logits = q2t_logits

        if self.training:
            logit_scale = self.clip.logit_scale.exp()
            retrieve_logits = logit_scale * retrieve_logits
            return retrieve_logits
        else:
            return retrieve_logits
