import os
import json
import glob
import numpy as np
import random
import heapq

import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import WEIGHTS_NAME, CONFIG_NAME, AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm

from calmV0820.drrn.conditioning_model import EthicsModel

def process(data_sample):
    """
    Given a data sample (dict), build a list of token_ids and act_mask
    """
    state_tokens = data_sample['state_tokens']
    # Add [CLS] to the head
    if state_tokens[0] != 50257:
        state_tokens = [50257] + state_tokens
    # Add [SEP] to the tail
    if state_tokens[-1] != 50258:
        state_tokens.append(50258)
    # Add [SEP] to the tail
    action_tokens = data_sample['action_tokens']
    if action_tokens[-1] != 50258:
        action_tokens.append(50258)
    # Build the outputs
    token_ids = state_tokens + action_tokens
    act_mask = np.concatenate([np.zeros(len(state_tokens)), np.ones(len(action_tokens))])
    moral_value = data_sample['moral_value']
    true_moral = data_sample['true_moral']

    return token_ids, act_mask, moral_value, true_moral


def process_act(data_sample):
    action_tokens = data_sample['action_tokens']
    if action_tokens[0] != 50257:
        action_tokens = [50257] + action_tokens
    if action_tokens[-1] != 50258:
        action_tokens.append(50258)
    act_mask = np.concatenate([np.ones(len(action_tokens))])
    moral_value = data_sample['moral_value']
    true_moral = data_sample['true_moral']

    return action_tokens, act_mask, moral_value, true_moral
    



def pad_sequences(data, pad_length, dtype):
    padded_data = np.zeros((len(data), pad_length), dtype=dtype)
    for i, line in enumerate(data):
        if len(line) > pad_length:
            line = line[len(line) - pad_length:]
        padded_data[i,:len(line)] = line
    return padded_data


class Trajectory(object):
    def __init__(self, transitions, score, test=False):
        if test:
            self.data_samples = transitions
        else:
            self.data_samples = self._build_data_samples(transitions)
        self.neg_length = -len(self.data_samples)
        self.score = score

    def __lt__(self, other):
        """
        First pop those with smaller score, then pop those longer
        """
        if self.score != other.score:
            return self.score < other.score
        else:
            return self.neg_length < other.neg_length

    def _build_data_samples(self, transitions):
        """
        Given a list of transitions, return a list of data_samples
        """
        data_samples = []
        for transition in transitions:
            data_samples.append({'state_tokens': transition.state.state,'action_tokens': transition.act, 'moral_value': transition.moral_value, 'true_moral': transition.true_moral})
        return data_samples


class LM_FineTuner(object):
    def __init__(self, lm, args):
        self.lm = lm

        self.lm_ft_freq = args.lm_ft_freq
        self.lm_ft_stop_at = args.lm_ft_stop_at
        self.lm_ft_epoch = args.lm_ft_epoch
        self.lm_ft_buffer_size = args.lm_ft_buffer_size
        self.lm_ft_buffer_min_start = args.lm_ft_buffer_min_start
        self.lm_ft_batch_size = args.lm_ft_batch_size

        self.lm_ft_loss_type = args.lm_ft_loss_type
        self.loss_add_weight = args.loss_add_weight
        self.loss_multi_weight = args.loss_multi_weight

        # Init buffer
        self.lm_ft_buffer = []
        self.lm_ft_thres_type = args.lm_ft_thres_type
        assert self.lm_ft_thres_type in {'max', 'mean', 'pos'}, 'Unsupported lm_ft_thres_type {}'.format(self.lm_ft_thres_type)
        self.score_threshold = 0
        # max -> add when the new score >= the largest score off the buffer
        # mean -> add when the new score >= the mean score of the buffer
        # pos -> add when the new score > 0       
        self.cclm = EthicsModel(args.cclm_model, args.cclm_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        

    def push(self, trajectory):
        """
        Push a trajectory to the buffer, and update the threshold
        """
        heapq.heappush(self.lm_ft_buffer, trajectory)
        print("Traj score {}, threshold {}, push {} samples! Now buffer has {} trajs".format(trajectory.score, 
                                                                                             self.score_threshold, 
                                                                                             -trajectory.neg_length, 
                                                                                             len(self.lm_ft_buffer)))
        if len(self.lm_ft_buffer) > self.lm_ft_buffer_size:
            to_remove = heapq.heappop(self.lm_ft_buffer)
            print("Pop traj: length {}, score {}".format(-to_remove.neg_length, to_remove.score))

        if self.lm_ft_thres_type == 'max':
            self.score_threshold = max(trajectory.score, self.score_threshold)      
        elif self.lm_ft_thres_type == 'mean':
            self.score_threshold = np.mean([item.score for item in self.lm_ft_buffer])
        # For the type "pos", the threshold is always 0
        elif self.lm_ft_thres_type == 'pos':
            self.score_threshold = 0


    def _build_finetune_dataloader(self):
        """
        Build a fine-tuning dataloader
        """
        print("Building FT dataset")
        
        token_id_set, act_mask_set, moral_value_set, true_moral_set = [], [], [], []
        for trajectory in self.lm_ft_buffer:
            for data_sample in trajectory.data_samples:
                token_ids, act_mask, moral_value, true_moral = process(data_sample)

                _ids, _mask, _, _ = process_act(data_sample)

                _ids = pad_sequences([_ids], 10, 'int')
                _mask = pad_sequences([_mask], 10, 'int')
                _ids = torch.cuda.LongTensor(_ids)
                _mask = torch.cuda.FloatTensor(_mask)
                _prob = self.cclm.get_prob(_ids, _mask)
                token_id_set.append(token_ids)
                act_mask_set.append(act_mask)
                if _prob > 0.5:
                    moral_value_set.append(moral_value)
                else:
                    moral_value_set.append(true_moral)

        att_mask_set = [np.ones(len(ids)) for ids in token_id_set]
        print("FT dataset contains {} samples".format(len(token_id_set)))

        # Limit the input to be <= 256
        token_ids = pad_sequences(token_id_set, 250, 'int')
        act_masks = pad_sequences(act_mask_set, 250, 'uint8')
        att_masks = pad_sequences(att_mask_set, 250, 'uint8')
        moral_values = moral_value_set
        true_morals = true_moral_set

        # Build the finetune dataloader
        ft_data = TensorDataset(torch.tensor(token_ids), 
                                torch.tensor(att_masks), 
                                torch.tensor(act_masks),
                                torch.tensor(moral_values),
                                torch.tensor(true_morals))
        ft_sampler = RandomSampler(ft_data)
        ft_dataloader = DataLoader(ft_data, 
                                   sampler=ft_sampler, 
                                   batch_size=self.lm_ft_batch_size, 
                                   drop_last=True)  # drop last batch for gpt-2
        return ft_dataloader


    def finetune(self):
        """
        Conduct fine-tuning, return the fine-tuning result (acc)
        """
        
        gradient_accumulation_steps = 1
        learning_rate = 2e-5
        adam_epsilon = 1e-8
        warmup_steps = .1
        weight_decay = 0
        max_grad_norm = 1.0
        ft_dataloader = self._build_finetune_dataloader()
        t_total = len(ft_dataloader) // gradient_accumulation_steps * self.lm_ft_epoch

        self.cclm.train_cclm(ft_dataloader)

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.lm.model.named_parameters() if not any(nd in n for nd in no_decay)], 
             'weight_decay': weight_decay},
            {'params': [p for n, p in self.lm.model.named_parameters() if any(nd in n for nd in no_decay)], 
             'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)

        global_step = 0
        model_to_resize = self.lm.model.module if hasattr(self.lm.model, 'module') else self.lm.model  # Take care of distributed/parallel training
        model_to_resize.resize_token_embeddings(len(self.lm.tokenizer))
        self.lm.model.zero_grad()
        ft_iterator = range(0, int(self.lm_ft_epoch))
        ft_losses, ft_accs = [], []

        for iter in ft_iterator:
            epoch_iterator = tqdm(ft_dataloader)
            total_actions = 0
            total_correct_actions = 0
            ft_loss = 0
            for step, batch in enumerate(epoch_iterator):
                b_input_ids, b_input_mask, b_strat, b_moral_values = batch
                b_labels = b_input_ids.clone()
                b_labels[b_strat == 0] = -100
                ground_truth = b_input_ids.clone()
                total_tokens_in_example = b_strat.sum(dim=1)
                b_input_ids = b_input_ids.to(self.device)
                b_labels = b_labels.to(self.device)
                b_input_mask = b_input_mask.to(self.device)
                b_moral_values = b_moral_values.to(self.device)

                self.lm.model.train()
                
                assert self.lm_ft_loss_type in {'multi', 'add', 'initial'}
                outputs = self.lm.model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels, 
                                moral_values = b_moral_values, loss_type = self.lm_ft_loss_type,loss_multi_weight = self.loss_multi_weight, 
                                loss_add_weight = self.loss_add_weight, iteration = iter)

                loss = outputs[0]
                if gradient_accumulation_steps > 1:
                    loss = loss / gradient_accumulation_steps
                loss.backward()
                loss_value = loss.item()
                ft_loss += loss_value

                prediction = torch.argmax(outputs[1], dim=2).to('cpu')
                pad = torch.zeros((prediction.shape[0], 1), dtype=torch.long)
                prediction = torch.cat((pad, prediction[:, :-1]), dim=1)
                diff = prediction - ground_truth == 0
                diff = diff * b_strat
                total_correct_for_each_example = diff.sum(dim=1)
                total_actions += b_input_ids.shape[0]
                total_correct_actions += (total_correct_for_each_example == total_tokens_in_example).sum()

                if (step + 1) % gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.lm.model.parameters(), max_grad_norm)
                    optimizer.step()
                    scheduler.step()  # Update learning rate schedule
                    self.lm.model.zero_grad()
                    global_step += 1

            ft_accs.append(total_correct_actions.item() / total_actions)
            ft_losses.append(ft_loss / total_actions)

            print("FT epoch {}/{}|Acc {:.3f}|Loss {:.3f}".format(iter+1, self.lm_ft_epoch, ft_accs[-1], ft_losses[-1]))
        
        

        return np.mean(ft_accs), np.mean(ft_losses)


    