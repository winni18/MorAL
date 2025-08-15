import os
import json
import glob
import numpy as np
import random
import heapq
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import WEIGHTS_NAME, CONFIG_NAME, AdamW, get_linear_schedule_with_warmup


def process(data_sample):
    state_tokens = data_sample['state_tokens']
    if not state_tokens or state_tokens[0] != 50257:
        state_tokens = [50257] + state_tokens
    if state_tokens[-1] != 50258:
        state_tokens.append(50258)

    action_tokens = data_sample['action_tokens']
    if not action_tokens or action_tokens[-1] != 50258:
        action_tokens = action_tokens + [50258]

    token_ids = state_tokens + action_tokens
    act_mask = np.concatenate([
        np.zeros(len(state_tokens), dtype=np.uint8),
        np.ones(len(action_tokens),  dtype=np.uint8)
    ])

    action_text = data_sample.get('action_text', '')   # ← 带出动作文本
    return token_ids, act_mask, action_text


def pad_sequences(data, pad_length, dtype):
    padded_data = np.zeros((len(data), pad_length), dtype=dtype)
    for i, line in enumerate(data):
        if len(line) > pad_length:
            line = line[len(line) - pad_length:]
        padded_data[i, :len(line)] = line
    return padded_data


def weight_from_cclm(cclm_reward, scheme="focal",
                     w_min=0.2, w_max=1.0,
                     gamma=2.0,        # focal
                     tau=0.5, temp=0.1,# sigmoid
                     margin=0.15       # piecewise
                     ):
    x = torch.as_tensor(cclm_reward, dtype=torch.float32,
                        device=(cclm_reward.device if torch.is_tensor(cclm_reward) else None))
    w_min_t = torch.tensor(float(w_min), dtype=torch.float32, device=x.device)
    w_max_t = torch.tensor(float(w_max), dtype=torch.float32, device=x.device)
    delta   = w_max_t - w_min_t

    if scheme == "focal":
        w = w_min_t + delta * torch.pow(1.0 - x, gamma)
    elif scheme == "sigmoid":
        z = (tau - x) / max(1e-6, float(temp))
        w = w_min_t + delta * torch.sigmoid(z)
    elif scheme == "piecewise":
        low, high = tau - margin, tau + margin
        w = torch.empty_like(x)
        left  = (x <= low)
        right = (x >= high)
        mid   = (~left) & (~right)
        w[left]  = w_max_t
        w[right] = w_min_t
        t = (x[mid] - low) / max(1e-6, (high - low))
        w[mid] = w_max_t * (1.0 - t) + w_min_t * t
    else:
        raise ValueError("Unknown scheme: {}".format(scheme))

    return torch.clamp(w, min=min(w_min, w_max), max=max(w_min, w_max))



def moral_policy_loss(shift_logits, shift_labels, cclm_reward,
                      i_iter,                  
                      c_scale=1.0,             
                      scheme="focal",          
                      w_min=0.2, w_max=1.0,
                      gamma=2.0, tau=0.5, temp=0.1, margin=0.15):

    B, T, V = shift_logits.shape
    device = shift_logits.device

    per_tok = F.cross_entropy(shift_logits.view(-1, V), shift_labels.view(-1), reduction='none', ignore_index=-100).view(B, T)

    valid = (shift_labels != -100).float()
    tok_cnt = valid.sum(dim=1).clamp_min(1.0)
    per_sample_ce = (per_tok * valid).sum(dim=1) / tok_cnt 

    if cclm_reward is None:
        cclm_reward = torch.zeros(B, dtype=torch.float32, device=device)
    else:
        cclm_reward = torch.as_tensor(cclm_reward, dtype=torch.float32, device=device)
    w = weight_from_cclm(cclm_reward, scheme=scheme, w_min=w_min, w_max=w_max,
                         gamma=gamma, tau=tau, temp=temp, margin=margin) 

    alpha = float(c_scale) * (1.0 - 0.05 * float(i_iter))
    if alpha < 0.0:
        alpha = 0.0

    # L_moral
    loss = alpha * (per_sample_ce * w).mean()
    return loss

class Trajectory(object):
    def __init__(self, transitions, score, test=False):
        if test:
            self.data_samples = transitions
        else:
            self.data_samples = self._build_data_samples(transitions)
        self.neg_length = -len(self.data_samples)
        self.score = score

    def __lt__(self, other):
        if self.score != other.score:
            return self.score < other.score
        else:
            return self.neg_length < other.neg_length

    def _build_data_samples(self, transitions):
        data_samples = []
        for tr in transitions:
            data_samples.append({
                'state_tokens': tr.state.state,
                'action_tokens': tr.act,
                'action_text': tr.act_string  
            })
        return data_samples


class LM_FineTuner(object):
    def __init__(self, lm, args, cclm=None):
        self.lm = lm
        self.lm_ft_freq = args.lm_ft_freq
        self.lm_ft_stop_at = args.lm_ft_stop_at
        self.lm_ft_epoch = args.lm_ft_epoch
        self.lm_ft_buffer_size = args.lm_ft_buffer_size
        self.lm_ft_buffer_min_start = args.lm_ft_buffer_min_start
        self.lm_ft_batch_size = args.lm_ft_batch_size
        self.lm_ft_loss_type = args.lm_ft_loss_type
        self.lm_ft_thres_type = args.lm_ft_thres_type
        assert self.lm_ft_thres_type in {'max', 'mean', 'pos'}, \
            f'Unsupported lm_ft_thres_type {self.lm_ft_thres_type}'
        
        self.cclm_weight_scheme = str(args.cclm_weight_scheme).lower()
        assert self.cclm_weight_scheme in {'focal', 'sigmoid', 'piecewise'}, \
            f'Unsupported cclm_weight_scheme: {self.cclm_weight_scheme}. '

        self.cclm_c_scale = args.cclm_c_scale
        self.cclm_w_min   = args.cclm_w_min
        self.cclm_w_max   = args.cclm_w_max
        self.cclm_gamma   = args.cclm_gamma
        self.cclm_tau     = args.cclm_tau
        self.cclm_temp    = args.cclm_temp
        self.cclm_margin  = args.cclm_margin
        self.cclm = cclm 
        self.cond_threshold = args.cond_threshold
        
        # Init buffer
        self.lm_ft_buffer = []
        self.lm_ft_thres_type = args.lm_ft_thres_type
        self.score_threshold = 0

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def push(self, trajectory):
        """
        Push a trajectory to the buffer, and update the threshold
        """
        heapq.heappush(self.lm_ft_buffer, trajectory)
        print("Traj score {}, threshold {}, push {} samples! Now buffer has {} trajs".format(
            trajectory.score, self.score_threshold, -trajectory.neg_length, len(self.lm_ft_buffer)))
        if len(self.lm_ft_buffer) > self.lm_ft_buffer_size:
            to_remove = heapq.heappop(self.lm_ft_buffer)
            print("Pop traj: length {}, score {}".format(-to_remove.neg_length, to_remove.score))

        if self.lm_ft_thres_type == 'max':
            self.score_threshold = max(trajectory.score, self.score_threshold)
        elif self.lm_ft_thres_type == 'mean':
            self.score_threshold = float(np.mean([item.score for item in self.lm_ft_buffer]))
        elif self.lm_ft_thres_type == 'pos':
            self.score_threshold = 0

    def _build_finetune_dataloader(self):
        print("Building FT dataset")
        token_id_set, act_mask_set, action_texts = [], [], []
        for traj in self.lm_ft_buffer:
            for data_sample in traj.data_samples:
                token_ids, act_mask, action_text = process(data_sample)
                token_id_set.append(token_ids)
                act_mask_set.append(act_mask)
                action_texts.append(action_text)

        att_mask_set = [np.ones(len(ids), dtype=np.uint8) for ids in token_id_set]
        print("FT dataset contains {} samples".format(len(token_id_set)))

        if len(token_id_set) == 0:
            empty = TensorDataset(torch.empty(0, 1, dtype=torch.long),
                                torch.empty(0, 1, dtype=torch.long),
                                torch.empty(0, 1, dtype=torch.long),
                                torch.empty(0, dtype=torch.float32))
            return DataLoader(empty, batch_size=self.lm_ft_batch_size)

        if self.cclm is not None and len(action_texts) > 0:
            try:
                batched = [[a] for a in action_texts]
                probs = self.cclm.get_probs(None, batched, self.cond_threshold)
                if torch.is_tensor(probs):
                    probs = probs.detach().cpu().numpy().tolist()

                cclm_rewards = []
                for p in probs:
                    if isinstance(p, (list, tuple)) and len(p) > 0:
                        cclm_rewards.append(float(p[0]))
                    else:
                        cclm_rewards.append(float(p))
            except Exception as e:
                print(f"[CCLM] batch get_probs failed: {e}")
                cclm_rewards = [0.0] * len(action_texts)
        else:
            cclm_rewards = [0.0] * len(action_texts)

        max_len = 250
        token_ids = pad_sequences(token_id_set, max_len, 'int64')
        act_masks = pad_sequences(act_mask_set, max_len, 'int64')
        att_masks = pad_sequences(att_mask_set, max_len, 'int64')

        ft_data = TensorDataset(
            torch.from_numpy(token_ids).long(),
            torch.from_numpy(att_masks).long(),     # attention_mask
            torch.from_numpy(act_masks).long(),     # strategy mask
            torch.tensor(cclm_rewards, dtype=torch.float32)
        )
        ft_sampler = RandomSampler(ft_data)
        ft_dataloader = DataLoader(
            ft_data,
            sampler=ft_sampler,
            batch_size=self.lm_ft_batch_size,
            drop_last=True,
            pin_memory=torch.cuda.is_available(),
            num_workers=0
        )
        return ft_dataloader


    def finetune(self):
        """
        Conduct fine-tuning, return the fine-tuning result (acc)
        """
        gradient_accumulation_steps = 1
        learning_rate = 2e-5
        adam_epsilon = 1e-8
        warmup_ratio = 0.1
        weight_decay = 0.0
        max_grad_norm = 1.0

        ft_dataloader = self._build_finetune_dataloader()
        steps_per_epoch = len(ft_dataloader) // max(1, gradient_accumulation_steps)
        t_total = steps_per_epoch * int(self.lm_ft_epoch)
        if t_total <= 0:
            print("No data for fine-tuning. Skip.")
            return 0.0, 0.0

        num_warmup_steps = int(t_total * warmup_ratio)
        if num_warmup_steps < 0:
            num_warmup_steps = 0
        if num_warmup_steps > t_total:
            num_warmup_steps = t_total

        # optimizer & scheduler
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.lm.model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': weight_decay},
            {'params': [p for n, p in self.lm.model.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=num_warmup_steps,
                                                    num_training_steps=t_total)

        model_to_resize = self.lm.model.module if hasattr(self.lm.model, 'module') else self.lm.model
        model_to_resize.resize_token_embeddings(len(self.lm.tokenizer))

        self.lm.model.train()
        self.lm.model.zero_grad()

        ft_losses, ft_accs = [], []

        for epoch in range(int(self.lm_ft_epoch)):
            epoch_iterator = tqdm(ft_dataloader, desc=f"FT epoch {epoch+1}/{self.lm_ft_epoch}")
            total_actions = 0
            total_correct_actions = 0
            epoch_loss_sum = 0.0

            for step, batch in enumerate(epoch_iterator):
                b_input_ids, b_input_mask, b_strat, b_rewards = batch

                b_labels = b_input_ids.clone()
                b_labels[b_strat == 0] = -100

                # for accuracy
                ground_truth = b_input_ids.clone()
                total_tokens_in_example = b_strat.sum(dim=1)

                # to device
                b_input_ids    = b_input_ids.to(self.device)
                b_input_mask   = b_input_mask.to(self.device)
                b_labels       = b_labels.to(self.device)
                b_rewards      = b_rewards.to(self.device)

                outputs = self.lm.model(
                    input_ids=b_input_ids,
                    attention_mask=b_input_mask,
                    token_type_ids=None,
                    labels=None
                )
                logits = outputs.logits  # [B,T,V]

                
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = b_labels[..., 1:].contiguous()
                
                loss = moral_policy_loss(
                    shift_logits, shift_labels,
                    cclm_reward=b_rewards,
                    i_iter=epoch,
                    c_scale=self.cclm_c_scale,
                    scheme=self.cclm_weight_scheme,
                    w_min=self.cclm_w_min, w_max=self.cclm_w_max,
                    gamma=self.cclm_gamma, tau=self.cclm_tau, temp=self.cclm_temp, margin=self.cclm_margin
                )

                if gradient_accumulation_steps > 1:
                    loss = loss / gradient_accumulation_steps

                loss.backward()
                epoch_loss_sum += float(loss.item()) * b_input_ids.size(0)

                with torch.no_grad():
                    prediction = torch.argmax(logits, dim=2)          
                    pad = torch.zeros((prediction.size(0), 1), dtype=torch.long, device=prediction.device)
                    prediction_shifted = torch.cat((pad, prediction[:, :-1]), dim=1)

                    pred_cpu = prediction_shifted.detach().cpu()
                    gt_cpu   = ground_truth.cpu()
                    strat_cpu= b_strat.cpu()

                    diff = (pred_cpu == gt_cpu)
                    diff = diff * strat_cpu
                    total_correct_for_each_example = diff.sum(dim=1)
                    total_actions += b_input_ids.size(0)
                    total_correct_actions += (total_correct_for_each_example == total_tokens_in_example.cpu()).sum().item()

                if (step + 1) % gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.lm.model.parameters(), max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    self.lm.model.zero_grad()

            ft_accs.append(total_correct_actions / float(total_actions))
            ft_losses.append(epoch_loss_sum / float(total_actions))
            print("FT epoch {}/{} | Acc {:.3f} | Loss {:.3f}".format(
                epoch + 1, self.lm_ft_epoch, ft_accs[-1], ft_losses[-1]
            ))

        return float(np.mean(ft_accs)), float(np.mean(ft_losses))
