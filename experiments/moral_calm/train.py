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
from conditioning_model import EthicsModel

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


def train(agent, lm_finetuner, envs, max_steps, update_freq, eval_freq, checkpoint_freq, log_freq, args, cclm=None):
    start = time.time()
    obs, rewards, dones, infos, transitions = [], [], [], [], []
    cumulative_morality_vector = np.zeros(4, dtype=np.int64)   
    env_steps, max_score, d_in, d_out = 0, 0, defaultdict(list), defaultdict(list)

    detector, reject = None, None
    if args.lm_top_k and args.lm_top_k > 0:
        try:
            import fasttext
            detector = fasttext.load_model('.valid_model.bin')  
            reject = lambda ob: detector.predict(clean(ob))[0][0] == '__label__invalid'
        except Exception as e:
            print(f'[fastText] load failed: {e}')
            detector, reject = None, None


    for env in envs:
        ob, info = env.reset()
        obs.append(ob); rewards.append(0); dones.append(False); infos.append(info); transitions.append([])

    states = build_state(lm_finetuner.lm, obs, infos, envs)
    valid_ids = [info['valid'] for info in infos]
    valid_ids_tokenized = [[lm_finetuner.lm.act2ids(a) for a in info['valid']] for info in infos]

    for step in range(1, max_steps + 1):
        action_ids, act_strings, action_idxs, action_values, cond_probs = agent.act(
            states, valid_ids_tokenized, valid_ids, lm=lm_finetuner.lm,
            eps=args.eps, alpha=args.alpha, k=args.eps_top_k, argmax=args.argmax, args=args)

        action_strs = [info['valid'][idx] for info, idx in zip(infos, action_idxs)]

        try:
            log('State  {}: {}'.format(step, lm_finetuner.lm.tokenizer.decode(states[0].state)))
            log('Actions{}: {}'.format(step, infos[0]['valid']))
            log('Qvalues{}: {}'.format(step, [round(v, 2) for v in action_values[0].tolist()]))
            log('CondProbs{}: {}'.format(step, [round(v, 2) for v in cond_probs[0].tolist()]))
        except Exception:
            pass

        next_obs, next_rewards, next_dones, next_infos = [], [], [], []
        for i, (env, action) in enumerate(zip(envs, action_strs)):
            if dones[i]:
                print(f"----- Step {step}, finish traj {i}")
                if (infos[i]['score'] > 0) and (infos[i]['score'] >= lm_finetuner.score_threshold):
                    lm_finetuner.push(Trajectory(transitions[i], infos[i]['score'], test=False))
                else:
                    print(f"Traj score {infos[i]['score']}, threshold {lm_finetuner.score_threshold}, do not push!")

                if env.max_score >= max_score:
                    for transition in transitions[i]:
                        agent.observe(transition, is_prior=True)

                env_steps += infos[i]['moves']
                env.env._write_to_log('<!DONE!>', 'step_log.txt')
                ob, info = env.reset()
                action_strs[i], action_ids[i], transitions[i] = 'reset', [], []
                next_obs += [ob]; next_rewards += [0]; next_dones += [False]; next_infos += [info]
                continue

            prev_inv, prev_look = infos[i]['inv'], infos[i]['look']
            ob, reward, done, info = env.step(action, write_to_log=True)
            env.env._write_to_log('STEP', 'step_log.txt')

            markers = info.get('markers', [])
            label_vectors = info.get('label_vectors', [])
            if i == 0:
                log(f'>> Action{step}: {action}')
                log(f"Reward{step}: {reward}, Score {info['score']}, Done {done}, Markers: {markers}, Label_vectors: {label_vectors}")
            for vec in label_vectors:
                cumulative_morality_vector += vec

            # 拒绝采样（可选）
            if reject is not None and args.lm_top_k and args.lm_top_k > 0:
                # 用 raw_state 作为 key；若缺失则降级
                try:
                    raw0, raw1, raw2 = states[i].raw_state
                    key = hash(tuple(raw0 + raw1 + raw2))
                except Exception:
                    key = hash(tuple(lm_finetuner.lm.sent2ids(str(states[i].state))))
                l_in, l_out = d_in[key], d_out[key]
                actions = list(infos[i]['valid'])
                actions_sorted = False
                rej = reject(ob) and prev_inv == info['inv'] and prev_look == info['look']

                while not done and rej and len(actions) > 1:
                    if not actions_sorted:
                        actions = [x for _, x in sorted(zip(action_values[i].tolist(), actions))]
                        actions.reverse(); actions_sorted = True
                    if action not in l_out: l_out.append(action)
                    if action in actions: actions.remove(action)
                    action = actions[0] if actions else action

                    ob, reward, done, info = env.step(action, write_to_log=True)
                    env.env._write_to_log('', 'step_log.txt')
                    if i == 0:
                        log(f'>> Action{step}: {action}')
                        log(f"Reward{step}: {reward}, Score {info['score']}, Done {done}, Markers: {info.get('markers', [])}, Label_vectors: {info.get('label_vectors', [])}")
                    rej = reject(ob) and prev_inv == info['inv'] and prev_look == info['look']
                action_strs[i] = action
                if (not rej) and (action not in l_in): l_in.append(action)
                if reward < 0 and action not in l_out: l_out.append(action)

            next_obs += [ob]; next_rewards += [reward]; next_dones += [done]; next_infos += [info]
            if info['score'] > max_score:
                max_score = info['score']; agent.memory.clear_alpha()
            if done: tb.logkv_mean('EpisodeScore', info['score'])

        rewards, dones, infos = next_rewards, next_dones, next_infos

        # 下一状态
        if args.lm_state_length_threshold > 0:
            next_states = build_state_with_threshold(lm_finetuner.lm, next_obs, infos, envs, obs, action_strs, args.lm_state_length_threshold)
        else:
            next_states = build_state(lm_finetuner.lm, next_obs, infos, envs, prev_obs=obs, prev_acts=action_strs)

        if args.lm_top_k and args.lm_top_k > 0:
            batched_states = [s.raw_state for s in next_states]
            batched_actions = lm_finetuner.lm.batched_generate(batched_states, k=args.lm_top_k, batch_size=args.lm_batch_size)
            for env, info, state, actions, done in zip(envs, infos, next_states, batched_actions, dones):
                if not done:
                    assert args.lm_type == 'gpt'
                    try:
                        raw0, raw1, raw2 = state.raw_state
                        key = hash(tuple(raw0 + raw1 + raw2))
                    except Exception:
                        key = hash(tuple(lm_finetuner.lm.sent2ids(str(state.state))))
                    l_in, l_out = d_in[key], d_out[key]
                    actions += [a for a in l_in if a not in actions]
                    actions = [a for a in actions if a and a not in l_out]
                    if not actions: actions = ['wait', 'yes', 'no']
                    info['valid'] = actions

        next_valids = [info['valid'] for info in infos]
        next_valids_tokenized = [[lm_finetuner.lm.act2ids(a) for a in info['valid']] for info in infos]

        for state, act, act_string, rew, next_state, valids, poss_acts, done, transition in \
                zip(states, action_ids, act_strings, rewards, next_states, next_valids_tokenized, next_valids, dones, transitions):
            if act:
                transition.append(Transition(state, act, act_string, rew, next_state, valids, poss_acts.copy(), done, 0.0))
                agent.observe(transition[-1])

        obs, states, valid_ids, valid_ids_tokenized = next_obs, next_states, next_valids, next_valids_tokenized

        # 日志
        if step % log_freq == 0:
            tb.logkv('Step', env_steps)
            tb.logkv("FPS", round((step * args.num_envs) / (time.time() - start), 2))
            tb.logkv("Max score seen", max_score)
            tb.logkv("#dict", len(lm_finetuner.lm.generate_dict))
            tb.logkv("Last50EpisodeScores", sum(env.get_end_scores(last=50) for env in envs) / len(envs))
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
            json.dump(d_in, open(f'{args.output_dir}/d_in.json', 'w'), indent=4)
            json.dump(d_out, open(f'{args.output_dir}/d_out.json', 'w'), indent=4)
            json.dump(lm_finetuner.lm.generate_dict, open(f'{args.output_dir}/lm.json', 'w'), indent=4)

        if (step % lm_finetuner.lm_ft_freq == 0) and (step < lm_finetuner.lm_ft_stop_at) and \
           (len(lm_finetuner.lm_ft_buffer) >= lm_finetuner.lm_ft_buffer_min_start):
            print(f"===== Step {step}, conduct FT and Val")
            lm_ft_acc, lm_ft_loss = lm_finetuner.finetune()
            tb.logkv('LMFT Acc', lm_ft_acc)
            tb.logkv('LMFT Loss', lm_ft_loss)
            print(f"FT Acc {lm_ft_acc:.3f}|FT Loss {lm_ft_loss:.3f}")

        if step % args.save_freq == 0:
            agent.save_model(args.output_dir)

        if (step == 5000) and (max_score == env.env.get_starting_score()):
            print('\n\nStopping training early due to lack of progress.\n\n')
            break

    # FINAL SAVES（同样加边界判断）
    tb.logkv('Step', env_steps)
    tb.logkv("FPS", round((step * args.num_envs) / (time.time() - start), 2))
    tb.logkv("Max score seen", max_score)
    tb.logkv("#dict", len(lm_finetuner.lm.generate_dict))
    tb.logkv("Last50EpisodeScores", sum(env.get_end_scores(last=50) for env in envs) / len(envs))
    tb.dumpkvs()

    json.dump(d_in, open(f'{args.output_dir}/d_in.json', 'w'), indent=4)
    json.dump(d_out, open(f'{args.output_dir}/d_out.json', 'w'), indent=4)
    json.dump(lm_finetuner.lm.generate_dict, open(f'{args.output_dir}/lm.json', 'w'), indent=4)
    agent.save_model(args.output_dir)
    print('TOTAL TIME ELAPSED:', time.time() - start)


def parse_args():
    import argparse
    parser = argparse.ArgumentParser()

    g = parser.add_argument_group("General")
    g.add_argument('--output_dir', default='./logs/zork1/ft_start0', type=str)
    g.add_argument('--load_dir', default=None, type=str)
    g.add_argument('--seed', default=1, type=int)

    g.add_argument('--game_folder_path', default='../annotated_games/zork1', type=str)
    g.add_argument('--env_step_limit', default=100, type=int)
    g.add_argument('--starting_percentage', default=0, type=int)
    g.add_argument('--num_envs', default=8, type=int)
    g.add_argument('--max_steps', default=35000, type=int)
    g.add_argument('--tensorboard', default=0, type=int)
    g.add_argument('--wandb', default=0, type=int)
    g.add_argument('--wandb_project', default='textgame', type=str)
    g.add_argument('--log_freq', default=100, type=int)
    g.add_argument('--checkpoint_freq', default=1000, type=int)
    g.add_argument('--eval_freq', default=5000, type=int)
    g.add_argument('--save_freq', default=10000, type=int)

    g.add_argument('--memory_size', default=10000, type=int)
    g.add_argument('--priority_fraction', default=0.5, type=float)
    g.add_argument('--batch_size', default=64, type=int)
    g.add_argument('--gamma', default=.9, type=float)
    g.add_argument('--learning_rate', default=0.0001, type=float)
    g.add_argument('--clip', default=5, type=float)
    g.add_argument('--embedding_dim', default=128, type=int)
    g.add_argument('--hidden_dim', default=128, type=int)

    g.add_argument('--update_freq', default=1, type=int)
    g.add_argument('--eps', default=None, type=float,
                    help='None: ~ softmax act_value; else eps-greedy exploration')
    g.add_argument('--eps_top_k', default=-1, type=int,
                    help='-1: uniform exploration; 0: ~ softmax lm_value; >0: ~ uniform(top-k by lm_value)')
    g.add_argument('--alpha', default=0.0, type=float,
                    help='act_value = alpha * bert_value + (1-alpha) * q_value (when eps is None)')
    g.add_argument('--argmax', action='store_true')
    
    lm = parser.add_argument_group("Language Model")
    lm.add_argument('--lm_type', default='gpt', choices=['gpt', 'ngram'])
    lm.add_argument('--lm_path', default='./model_weights/gpt2', type=str)
    lm.add_argument('--fp16', action='store_true')
    lm.add_argument('--model_parallel', action='store_true')
    lm.add_argument('--lm_top_k', default=40, type=int)
    lm.add_argument('--lm_batch_size', default=None, type=int)
    lm.add_argument('--lm_state_length_threshold', default=250, type=int)
    lm.add_argument('--fasttext_valid_model', default=None, type=str)

    cm = parser.add_argument_group("Ethics Model")
    cm.add_argument('--cclm_path', default='./ethics/cm_roberta-large.pt', type=str)
    cm.add_argument('--conditioning_model_load_dir', default=None, type=str)
    cm.add_argument('--cclm_model', default='roberta-large', type=str)
    cm.add_argument('--cclm_oracle', action='store_true')
    cm.add_argument('--cond_threshold', default=0.5, type=float)

    ft = parser.add_argument_group("LM Finetune")
    ft.add_argument('--lm_ft_loss_type', default='max', choices=['max', 'mean', 'pos'])
    ft.add_argument('--lm_ft_freq', default=2000, type=int)
    ft.add_argument('--lm_ft_stop_at', default=10_000_000, type=int)
    ft.add_argument('--lm_ft_buffer_size', default=50, type=int)
    ft.add_argument('--lm_ft_buffer_min_start', default=15, type=int)
    ft.add_argument('--lm_ft_thres_type', default='max', choices=['max', 'mean', 'pos'])
    ft.add_argument('--lm_ft_epoch', default=3, type=int)
    ft.add_argument('--lm_ft_batch_size', default=4, type=int)
    ft.add_argument('--cclm_weight_scheme', default='focal', choices=['focal', 'sigmoid', 'piecewise'])
    ft.add_argument('--cclm_c_scale', default=1.0, type=float)
    ft.add_argument('--cclm_w_min', default=0.2, type=float)
    ft.add_argument('--cclm_w_max', default=1.0, type=float)
    ft.add_argument('--cclm_gamma', default=2.0, type=float)
    ft.add_argument('--cclm_tau', default=0.5, type=float)
    ft.add_argument('--cclm_temp', default=0.1, type=float)
    ft.add_argument('--cclm_margin', default=0.15, type=float)

    args = parser.parse_args()

    return args





def main():
    start_time = time.time()

    assert jericho.__version__.startswith('3'), "This code is designed to be run with Jericho version >= 3.0.0."
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print('============== RUNNING TRAINING FOR {} ==============\n'.format(args.game_folder_path))

    if args.cclm_path:
        cclm = EthicsModel(
            args.cclm_model,
            args.cclm_path,
            args.conditioning_model_load_dir or args.load_dir,
            oracle=args.cclm_oracle
        )
    else:
        cclm = None

    language_model = GPT2LM(args.lm_path, args.fp16, model_parallel=args.model_parallel)
    lm_finetuner = LM_FineTuner(lm=language_model, args=args, cclm=cclm)

    args.vocab_size = len(lm_finetuner.lm.tokenizer)
    print(args)

    configure_logger(args.output_dir, args.tensorboard, args.wandb, args)
    agent = DRRN_Agent(args)

    envs = [EnvCALM(game_folder_path=args.game_folder_path, step_limit=args.env_step_limit, seed=args.seed,
                    starting_percentage=args.starting_percentage,
                    log_dir=os.path.join(args.output_dir, 'log_files', f'env{i}'))
            for i in range(args.num_envs)]

    print('MAX SCORE FOR GAME: {}'.format(envs[0].env.get_max_score()))
    print('STARTING SCORE: {}'.format(envs[0].env.get_starting_score()))
    print('\n==============================================\n')

    train(agent, lm_finetuner, envs, args.max_steps, args.update_freq,
          args.eval_freq, args.checkpoint_freq, args.log_freq, args, cclm=cclm)

    with open(os.path.join(args.output_dir, 'experiment_info.pkl'), 'wb') as f:
        pickle.dump({'status': 'Done', 'time': time.time() - start_time,
                     'peak_memory': torch.cuda.max_memory_allocated()}, f)



if __name__ == "__main__":
	main()
