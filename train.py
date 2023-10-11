# [20200810]
# finetune calm and modify loss function (in /custom_transformers/models/gpt2/modeling_gpt2.py)
import os
import time
import jericho
import logger
import argparse
import logging
import json
from jericho.util import clean
from random import choice
from collections import defaultdict
import numpy as np
import pickle

from env_calm import EnvCALM

from lm import *
from drrn import *
from ft_calm import Trajectory, LM_FineTuner

logging.getLogger().setLevel(logging.CRITICAL)


def configure_logger(log_dir, add_tb=1, add_wb=1, args=None):
	logger.configure(log_dir, format_strs=['log'])
	global tb
	log_types = [logger.make_output_format('json', log_dir),
				 logger.make_output_format('stdout', log_dir)]
	if add_tb: log_types += [logger.make_output_format('tensorboard', log_dir)]
	if add_wb: log_types += [logger.make_output_format('wandb', log_dir, args=args)]
	tb = logger.Logger(log_dir, log_types)
	global log
	log = logger.log


def train(agent, lm_finetuner, envs, max_steps, update_freq, eval_freq, checkpoint_freq, log_freq, args):
	start = time.time()
	obs, rewards, dones, infos, transitions, moral_values, true_morals= [], [], [], [], [], [], []
	cumulative_morality_vector = np.zeros(4, dtype=np.int)
	moral_per_episode = [] # moral value per episode
	env_steps, max_score, d_in, d_out = 0, 0, defaultdict(list), defaultdict(list)
	
	if args.lm_top_k:  # small pre-trained classifier to filter invalid actions that CALM generates
		import fasttext
		detector = fasttext.load_model('./valid_model.bin')
		reject = lambda ob: detector.predict(clean(ob))[0][0] == '__label__invalid'
	
	assert not args.lm_dict
	# if args.lm_dict:
	# 	print('LOADING LM GENERATION DICT')
	# 	g_dict = json.load(open(args.lm_dict, 'r'))
	# 	lm.generate_dict = {int(k):v for k,v in g_dict.items()}

	for env in envs:
		ob, info = env.reset()
		obs, rewards, dones, infos, transitions, moral_values, true_morals = \
			obs + [ob], rewards + [0], dones + [False], infos + [info], transitions + [[]], moral_values + [0], true_morals + [0]
	states = build_state(lm_finetuner.lm, obs, infos, envs)
	valid_ids = [info['valid'] for info in infos]
	valid_ids_tokenized = [[lm_finetuner.lm.act2ids(a) for a in info['valid']] for info in infos]

	for step in range(1, max_steps + 1):

		# if args.cclm_oracle:
		# 	assert cclm is not None
		# 	cclm.envs = envs  

		action_ids, act_strings, action_idxs, action_values, cond_probs = agent.act(states, valid_ids_tokenized, valid_ids, lm=lm_finetuner.lm,
						eps=args.eps, alpha=args.alpha, k=args.eps_top_k, argmax=args.argmax, args=args)			

		action_strs = [info['valid'][idx] for info, idx in zip(infos, action_idxs)]

		log('State  {}: {}'.format(step, lm_finetuner.lm.tokenizer.decode(states[0].state)))
		log('Actions{}: {}'.format(step, infos[0]['valid']))
		log('Qvalues{}: {}'.format(step, [round(value, 2) for value in action_values[0].tolist()]))
		log('CondProbs{}: {}'.format(step, [round(value, 2) for value in cond_probs[0].tolist()]))

		# step with rejection
		next_obs, next_rewards, next_dones, next_infos, next_moral_values, next_true_morals = [], [], [], [], [], []
		for i, (env, action) in enumerate(zip(envs, action_strs)):
			if dones[i]:

				print("----- Step {}, finish traj {}".format(step, i))
				if (infos[i]['score'] > 0) and (infos[i]['score'] >= lm_finetuner.score_threshold):
					lm_finetuner.push(Trajectory(transitions[i], infos[i]['score'], test=False))
				else:
					print("Traj score {}, threshold {}, do not push!".format(infos[i]['score'], lm_finetuner.score_threshold))

				if env.max_score >= max_score:  # put in alpha queue
					for transition in transitions[i]:
						agent.observe(transition, is_prior=True)
						
				env_steps += infos[i]['moves']
				moral_per_episode.append(cumulative_morality_vector[0])
				env.env._write_to_log('<!DONE!>', 'step_log.txt')
				ob, info = env.reset()
				action_strs[i], action_ids[i], transitions[i] = 'reset', [], []
				next_obs, next_rewards, next_dones, next_infos, next_moral_values, next_true_morals = next_obs + [ob], next_rewards + [0], next_dones + [
					False], next_infos + [info], next_moral_values + [0], next_true_morals + [0]
				continue

			prev_inv, prev_look = infos[i]['inv'], infos[i]['look']
			ob, reward, done, info = env.step(action, write_to_log=True)
			env.env._write_to_log('STEP', 'step_log.txt')  # first action for this step

			markers, label_vectors = info['markers'], info['label_vectors']



			if (info['label_vectors'] != []) and (sum(info['label_vectors'])[0] > 0):
				true_moral = 1
			else:
				true_moral = 0
			moral_value = lm_finetuner.cclm.get_probs(None, [[action]], args.cond_threshold)[0].item()
			# moral_value = moral_value - args.condition_weight * moral_value

			if i == 0:
				log('>> Action{}: {}'.format(step, action))
				log("Reward{}: {}, Score {}, Done {}, Markers: {}, Label_vectors: {}".format(step, reward, info['score'], done, markers, label_vectors))
			for vec in label_vectors:
				cumulative_morality_vector += vec

			if args.lm_top_k:  # deal with rejection
				key = hash(tuple(states[i][0] + states[i][1] + states[i][2]))
				l_in, l_out = d_in[key], d_out[key]
				actions = infos[i]['valid']
				actions_sorted = False
				rej = reject(ob) and prev_inv == info['inv'] and prev_look == info['look']

				# while action is invalid, pull another action from CALM generated candidates
				while not done and rej and len(actions) > 1:
					if not actions_sorted:
						actions = [x for _, x in sorted(zip(action_values[i].tolist(), actions))]
						actions.reverse() # descending
						actions_sorted = True

					if action not in l_out: l_out.append(action)
					actions.remove(action)
					# action = choice(actions)
					action = actions[0]

					ob, reward, done, info = env.step(action, write_to_log=True)
					env.env._write_to_log('', 'step_log.txt')  # alternate actions for this step
					markers, label_vectors = info['markers'], info['label_vectors']
					# 1

					if (info['label_vectors'] != []) and (sum(info['label_vectors'])[0] > 0):
						true_moral = 1
					else:
						true_moral = 0
					moral_value = lm_finetuner.cclm.get_probs(None, [[action]], args.cond_threshold)[0].item()
						# moral_value = moral_value - args.condition_weight * moral_value

					if i == 0:
						log('>> Action{}: {}'.format(step, action))
						log("Reward{}: {}, Score {}, Done {}, Markers: {}, Label_vectors: {}".format(step, reward, info['score'], done, markers, label_vectors))
					for vec in label_vectors:
						cumulative_morality_vector += vec

					rej = reject(ob) and prev_inv == info['inv'] and prev_look == info['look']
				action_strs[i] = action

				if not rej and action not in l_in: l_in.append(action)
				if reward < 0 and action not in l_out: l_out.append(action)  # screen negative-reward actions

			next_obs, next_rewards, next_dones, next_infos, next_moral_values, next_true_morals = \
				next_obs + [ob], next_rewards + [reward], next_dones + [done], next_infos + [info], next_moral_values + [moral_value], next_true_morals + [true_moral]
			if info['score'] > max_score:  # new high score experienced
				max_score = info['score']
				agent.memory.clear_alpha()
			if done: tb.logkv_mean('EpisodeScore', info['score'])
		rewards, dones, infos, moral_values, true_morals = next_rewards, next_dones, next_infos, next_moral_values, next_true_morals
		

		# ================= GENERATE VALID ACTIONS ================= #
		if args.lm_state_length_threshold > 0:
			next_states = build_state_with_threshold(lm_finetuner.lm, 
                                                     next_obs, 
                                                     infos, 
													 envs,
                                                     obs, 
                                                     action_strs,
                                                     args.lm_state_length_threshold)													 
		else:
			next_states = build_state(lm_finetuner.lm, next_obs, infos, envs, prev_obs=obs, prev_acts=action_strs)
		
		if args.lm_top_k:
			batched_states = [s.raw_state for s in next_states]
			batched_actions = lm_finetuner.lm.batched_generate(batched_states, k=args.lm_top_k, batch_size=args.lm_batch_size)
			# for env, info, state, done in zip(envs, infos, next_states, dones):
			for env, info, state, actions, done in zip(envs, infos, next_states, batched_actions, dones):
				if not done:
					assert args.lm_type == 'gpt'
					key = hash(tuple(state[0] + state[1] + state[2]))
					l_in, l_out = d_in[key], d_out[key]
					actions += [action for action in l_in if action not in actions]  # add extra valid
					actions = [action for action in actions if action and action not in l_out]  # remove invalid
					if not actions: actions = ['wait', 'yes', 'no']
					info['valid'] = actions
		next_valids = [info['valid'] for info in infos]
		next_valids_tokenized = [[lm_finetuner.lm.act2ids(a) for a in info['valid']] for info in infos]


		for state, act, act_string, rew, next_state, valids, poss_acts, done, moral_value, true_moral, transition in zip(states, action_ids, act_strings, rewards, next_states,
																		 next_valids_tokenized, next_valids, dones, moral_values, true_morals, transitions):
			if act:  # not [] (i.e. reset)
				transition.append(Transition(state, act, act_string, rew, next_state, valids, poss_acts.copy(), done, moral_value, true_moral))
				agent.observe(transition[-1])  # , is_prior=(rew != 0))
		obs, states, valid_ids, valid_ids_tokenized = next_obs, next_states, next_valids, next_valids_tokenized

		# if args.policy_shaping and args.cclm_oracle:  # this is for filling the cclm cache; needed for oracle implementation
		# 	_ = cclm.get_probs(states, valid_ids, cond_threshold=args.cond_threshold)

		if step % log_freq == 0:
			tb.logkv('Step', env_steps)
			tb.logkv("FPS", round((step * args.num_envs) / (time.time() - start), 2))
			tb.logkv("Max score seen", max_score)
			tb.logkv("ethics_value", (moral_per_episode[-1]-moral_per_episode[-2]))
			tb.logkv("#dict", len(lm_finetuner.lm.generate_dict))
			tb.logkv("Last50EpisodeScores", sum(env.get_end_scores(last=50) for env in envs) / len(envs))
			tb.dumpkvs()
			if len(moral_per_episode) > 51:
				tb.logkv("Last50Ethics", (moral_per_episode[-1]-moral_per_episode[-51]) / 50)
			if len(lm_finetuner.lm_ft_buffer) > 0:
				tb.logkv("LMFT BufferMaxScore", np.max([item.score for item in lm_finetuner.lm_ft_buffer]))
				tb.logkv("LMFT BufferMeanScore", np.mean([item.score for item in lm_finetuner.lm_ft_buffer]))
				tb.logkv("LMFT BufferMeanLength", -np.mean([item.neg_length for item in lm_finetuner.lm_ft_buffer]))
			tb.dumpkvs()
		
		if step % update_freq == 0:
			loss = agent.update(args=args)
			if loss is not None:
				tb.logkv_mean('Loss', loss)

		if step % checkpoint_freq == 0:
			json.dump(d_in, open('%s/d_in.json' % args.output_dir, 'w'), indent=4)
			json.dump(d_out, open('%s/d_out.json' % args.output_dir, 'w'), indent=4)
			json.dump(lm_finetuner.lm.generate_dict, open('%s/lm.json' % args.output_dir, 'w'), indent=4)

		if (step % lm_finetuner.lm_ft_freq == 0) and (step < lm_finetuner.lm_ft_stop_at) and (len(lm_finetuner.lm_ft_buffer) >= lm_finetuner.lm_ft_buffer_min_start):
			print("===== Step {}, conduct FT and Val".format(step))
			lm_ft_acc, lm_ft_loss = lm_finetuner.finetune()
			tb.logkv('LMFT Acc', lm_ft_acc)
			tb.logkv('LMFT Loss', lm_ft_loss)
			print("FT Acc {:.3f}|FT Loss {:.3f}".format(lm_ft_acc, lm_ft_loss))

		if step % args.save_freq == 0:
			# save agent model to output_dir
			agent.save_model(args.output_dir)

		if (step == 5000) and (max_score == env.env.get_starting_score()):
			print('\n\nStopping training early due to lack of progress.\n\n')
			break  # cut off training prematurely if there is no progress after 5000 steps
	
	# ============================ FINAL SAVES ============================ #
	tb.logkv('Step', env_steps)
	tb.logkv("FPS", round((step * args.num_envs) / (time.time() - start), 2))
	tb.logkv("Max score seen", max_score)
	tb.logkv("#dict", len(lm_finetuner.lm.generate_dict))
	tb.logkv("ethics_value", (moral_per_episode[-1]-moral_per_episode[-2]))
	tb.logkv("Last50Ethics", (moral_per_episode[-1]-moral_per_episode[-51]) / 50)
	tb.logkv("Last50EpisodeScores", sum(env.get_end_scores(last=50) for env in envs) / len(envs))
	tb.dumpkvs()

	json.dump(d_in, open('%s/d_in.json' % args.output_dir, 'w'), indent=4)
	json.dump(d_out, open('%s/d_out.json' % args.output_dir, 'w'), indent=4)
	json.dump(lm_finetuner.lm.generate_dict, open('%s/lm.json' % args.output_dir, 'w'), indent=4)

	agent.save_model(args.output_dir)

	print('TOTAL TIME ELAPSED:', time.time() - start)


def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--output_dir', default='./logs/zork1/ft_start0')
	parser.add_argument('--load_dir', default=None, type=str)
	parser.add_argument('--env_step_limit', default=100, type=int)
	parser.add_argument('--seed', default=1, type=int)
	parser.add_argument('--num_envs', default=8, type=int)
	parser.add_argument('--max_steps', default=35000, type=int)
	parser.add_argument('--update_freq', default=1, type=int)
	parser.add_argument('--save_freq', default=10000, type=int)
	parser.add_argument('--checkpoint_freq', default=1000, type=int)
	parser.add_argument('--eval_freq', default=5000, type=int)
	parser.add_argument('--log_freq', default=100, type=int)
	parser.add_argument('--memory_size', default=10000, type=int)
	parser.add_argument('--priority_fraction', default=0.5, type=float)
	parser.add_argument('--batch_size', default=64, type=int)
	parser.add_argument('--gamma', default=.9, type=float)
	parser.add_argument('--learning_rate', default=0.0001, type=float)
	parser.add_argument('--clip', default=5, type=float)
	parser.add_argument('--embedding_dim', default=128, type=int)
	parser.add_argument('--hidden_dim', default=128, type=int)

	# logger
	parser.add_argument('--tensorboard', default=0, type=int)
	parser.add_argument('--wandb', default=0, type=int)
	parser.add_argument('--wandb_project', default='textgame', type=str)

	# language model
	parser.add_argument('--lm_top_k', default=40, type=int,
						help='when >0, use lm top-k actions in place of jericho action detection')
	parser.add_argument('--lm_type', default='gpt', help='gpt | ngram')
	parser.add_argument('--lm_path', default='')
	parser.add_argument('--lm_dict', default='')
	parser.add_argument('--fp16', action='store_true')
	parser.add_argument('--lm_batch_size', default=None, type=int, help='batch size for CALM action generator; defaults to all envs at once')
	parser.add_argument('--model_parallel', action='store_true')

	# useless in our study
	parser.add_argument('--cclm_path', default='cm_roberta-large.pt', type=str)
	parser.add_argument('--cclm_model', default='roberta-large', type=str)
	# parser.add_argument('--conditioning_model_load_dir', default=None, type=str)
	parser.add_argument('--reward_shaping', action='store_false')
	# parser.add_argument('--policy_shaping', action='store_true')
	# parser.add_argument('--cclm_oracle', action='store_true')
	parser.add_argument('--condition_weight', default=0, type=float)
	parser.add_argument('--cond_threshold', default=0.39, type=float)
	# parser.add_argument('--weight_conditioning_by_max_score', action='store_true')
	# parser.add_argument('--adaptive_cond_weight', action='store_true')
	# parser.add_argument('--adaptivmoral_valueethie_cond_weight2', action='store_true')
	
	# game
	parser.add_argument('--game_folder_path', type=str, default='')

	# fast-forwarding
	parser.add_argument('--starting_percentage', type=int, default=0)

	parser.add_argument('--lm_state_length_threshold', default=250, type=int) # Whether or not to use the length threshold

	# LM finetune related
	parser.add_argument('--lm_ft_freq', default=1, type=int, help='fine-tuning frequency') # 2000
	parser.add_argument('--lm_ft_stop_at', default=10000000, type=int, help='when to stop fine-tuning, e.g. after 20000 steps') # 10000000
	parser.add_argument('--lm_ft_buffer_size', default=50, type=int, help='the FT buffer size') # 50
	parser.add_argument('--lm_ft_buffer_min_start', default=1, type=int, help='the min buffer size for starting FT') # 20
	parser.add_argument('--lm_ft_thres_type', default='max', type=str, help='how to determine the threshold for pushing, max / mean / pos') # max
	parser.add_argument('--lm_ft_epoch', default=3, type=int, help='the number of epochs per FT') # 3
	parser.add_argument('--lm_ft_batch_size', default=4, type=int, help='the FT batch size')      # 8

	# LM finetune loss
	parser.add_argument('--lm_ft_loss_type', default='multi', type=str,  
						help='multi: loss*weight*(moral);  add: loss + weight*moral*(1-itera*0.05);  initial: loss') # multi, add, initial
	parser.add_argument('--loss_add_weight', default=0.3, type=int)
	parser.add_argument('--loss_multi_weight', default=10, type=int)


	# exploration
	parser.add_argument('--eps', default=None, type=float,
						help='None: ~ softmax act_value; else eps-greedy-exploration')
	parser.add_argument('--eps_top_k', default=-1, type=int,
						help='-1: uniform exploration; 0: ~ softmax lm_value; >0: ~ uniform(top k w.r.t. lm_value)')
	parser.add_argument('--alpha', default=0, type=float,
						help='act_value = alpha * bert_value + (1-alpha) * q_value; only used when eps is None now')
	parser.add_argument('--argmax', action='store_true')

	return parser.parse_args()


def main():
	start_time = time.time()
        
	assert jericho.__version__.startswith('3'), "This code is designed to be run with Jericho version >= 3.0.0."
	args = parse_args()
	
	os.makedirs(args.output_dir, exist_ok=True)

	print('==============================================')
	print('============== RUNNING TRAINING FOR {} ==============\n'.format(args.game_folder_path))

	
	if (args.lm_dict != '') and (os.path.exists(args.lm_dict) == False):
		print('could not find lm_dict; defaulting to empty cache')
		args.lm_dict = ''

	language_model = GPT2LM(args.lm_path, args.fp16, model_parallel=args.model_parallel)
	lm_finetuner = LM_FineTuner(lm=language_model, args=args)


	args.vocab_size = len(lm_finetuner.lm.tokenizer)
	
	print(args)


	configure_logger(args.output_dir, args.tensorboard, args.wandb, args)
	agent = DRRN_Agent(args)

	# =========================== CREATE ENVS =========================== #
	envs = [EnvCALM(game_folder_path=args.game_folder_path, step_limit=args.env_step_limit, seed=args.seed,
						  starting_percentage=args.starting_percentage,
						  log_dir=os.path.join(args.output_dir, 'log_files', 'env{}'.format(i)))
			for i in range(args.num_envs)]
	

	# =========================== START TRAINING =========================== #
	print('MAX SCORE FOR GAME: {}'.format(envs[0].env.get_max_score()))
	print('STARTING SCORE: {}'.format(envs[0].env.get_starting_score()))

	print('\n==============================================\n')

	train(agent, lm_finetuner, envs, args.max_steps, args.update_freq, args.eval_freq, args.checkpoint_freq, args.log_freq, args)

	with open(os.path.join(args.output_dir, 'experiment_info.pkl'), 'wb') as f:
		pickle.dump({'status': 'Done', 'time': time.time() - start_time,
					 'peak_memory': torch.cuda.max_memory_allocated()}, f)


if __name__ == "__main__":
	main()
