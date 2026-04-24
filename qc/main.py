import glob, tqdm, wandb, os, json, random, time, jax
import jax.numpy as jnp
import numpy as np
import optax

from absl import app, flags
from ml_collections import config_flags

from log_utils import setup_wandb, get_exp_name, get_flag_dict, CsvLogger

from envs.env_utils import make_env_and_datasets
from envs.ogbench_utils import make_ogbench_env_and_datasets
from envs.robomimic_utils import is_robomimic_env

from utils.flax_utils import save_agent, load_agent
from utils.datasets import Dataset, ReplayBuffer

from evaluation import evaluate
from agents import agents


if 'CUDA_VISIBLE_DEVICES' in os.environ:
    os.environ['EGL_DEVICE_ID'] = os.environ['CUDA_VISIBLE_DEVICES']
    os.environ['MUJOCO_EGL_DEVICE_ID'] = os.environ['CUDA_VISIBLE_DEVICES']

FLAGS = flags.FLAGS

flags.DEFINE_string('run_group', 'Debug', 'Run group.')
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_string('env_name', 'cube-triple-play-singletask-task2-v0', 'Environment (dataset) name.')
flags.DEFINE_string('save_dir', 'exp/', 'Save directory.')

flags.DEFINE_integer('offline_steps', 50000, 'Number of offline steps.')
flags.DEFINE_integer('online_steps', 50000, 'Number of online steps.')
flags.DEFINE_integer('buffer_size', 2000000, 'Replay buffer size.')
flags.DEFINE_integer('log_interval', 5000, 'Logging interval.')
flags.DEFINE_integer('eval_interval', 5000, 'Evaluation interval.')
flags.DEFINE_integer('save_interval', -1, 'Save interval.')
flags.DEFINE_integer('start_training', 5000, 'when does training start')

flags.DEFINE_integer('utd_ratio', 1, "update to data ratio")
flags.DEFINE_float('discount', 0.99, 'discount factor')

flags.DEFINE_integer('eval_episodes', 10, 'Number of evaluation episodes.')
flags.DEFINE_integer('video_episodes', 0, 'Number of video episodes for each task.')
flags.DEFINE_integer('video_frame_skip', 3, 'Frame skip for videos.')

config_flags.DEFINE_config_file('agent', 'agents/acfql.py', lock_config=False)

flags.DEFINE_float('dataset_proportion', 1.0, "Proportion of the dataset to use")
flags.DEFINE_integer('dataset_replace_interval', 1000, 'Dataset replace interval')
flags.DEFINE_string('ogbench_dataset_dir', None, 'OGBench dataset directory')

flags.DEFINE_integer('horizon_length', 5, 'action chunking length.')
flags.DEFINE_bool('sparse', False, "make the task sparse reward")
flags.DEFINE_bool('save_all_online_states', False, "save all trajectories to npy")

# ==================== FLAGS CHO CHECKPOINT & DISCRIMINATOR ====================
flags.DEFINE_string('load_checkpoint', '', 'Đường dẫn tới folder offline_checkpoint để load')
flags.DEFINE_bool('save_offline', True, 'Có lưu agent sau khi xong pha offline hay không')

flags.DEFINE_bool('use_discriminator', False, "Enable success/failure discriminator shaping")
flags.DEFINE_float('disc_beta', 0.05, "Weight of discriminator reward")
flags.DEFINE_float('disc_decay_ratio', 0.3, "Phần trăm chặng đường online để beta giảm về 0")
flags.DEFINE_integer('disc_update_interval', 5000, "Train discriminator every N online steps")


class RunningMeanStd:
    def __init__(self, epsilon=1e-4):
        self.mean = 0.0
        self.var = 1.0
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean(x)
        batch_var = np.var(x)
        batch_count = x.shape[0]
        
        if batch_count == 0: return

        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        self.mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + (delta ** 2) * self.count * batch_count / tot_count
        self.var = M2 / tot_count
        self.count = tot_count


class LoggingHelper:
    def __init__(self, csv_loggers, wandb_logger):
        self.csv_loggers = csv_loggers
        self.wandb_logger = wandb_logger

    def log(self, data, prefix, step):
        assert prefix in self.csv_loggers, prefix
        self.csv_loggers[prefix].log(data, step=step)
        self.wandb_logger.log({f'{prefix}/{k}': v for k, v in data.items()}, step=step)


def main(_):
    exp_name = get_exp_name(FLAGS.seed)
    run = setup_wandb(project='qc', group=FLAGS.run_group, name=exp_name)
    
    FLAGS.save_dir = os.path.join(FLAGS.save_dir, wandb.run.project, FLAGS.run_group, FLAGS.env_name, exp_name)
    os.makedirs(FLAGS.save_dir, exist_ok=True)
    flag_dict = get_flag_dict()

    with open(os.path.join(FLAGS.save_dir, 'flags.json'), 'w') as f:
        json.dump(flag_dict, f)

    config = FLAGS.agent

    # ====================== DATA LOADING ======================
    if FLAGS.ogbench_dataset_dir is not None:
        assert FLAGS.dataset_replace_interval != 0
        assert FLAGS.dataset_proportion == 1.0
        dataset_idx = 0
        dataset_paths = [file for file in sorted(glob.glob(f"{FLAGS.ogbench_dataset_dir}/*.npz")) 
                        if '-val.npz' not in file]
        env, eval_env, train_dataset, val_dataset = make_ogbench_env_and_datasets(
            FLAGS.env_name, dataset_path=dataset_paths[dataset_idx], compact_dataset=False)
    else:
        env, eval_env, train_dataset, val_dataset = make_env_and_datasets(FLAGS.env_name)

    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)

    online_rng, rng = jax.random.split(jax.random.PRNGKey(FLAGS.seed), 2)
    log_step = 0
    discount = FLAGS.discount
    config["horizon_length"] = FLAGS.horizon_length

    def process_train_dataset(ds):
        ds = Dataset.create(**ds)
        if FLAGS.dataset_proportion < 1.0:
            new_size = int(len(ds['masks']) * FLAGS.dataset_proportion)
            ds = Dataset.create(**{k: v[:new_size] for k, v in ds.items()})
        
        if is_robomimic_env(FLAGS.env_name):
            ds_dict = {k: v for k, v in ds.items()}
            ds_dict["rewards"] = ds["rewards"] - 1.0
            ds = Dataset.create(**ds_dict)
        
        if FLAGS.sparse:
            ds_dict = {k: v for k, v in ds.items()}
            ds_dict["rewards"] = (ds["rewards"] != 0.0) * -1.0
            ds = Dataset.create(**ds_dict)
        return ds

    train_dataset = process_train_dataset(train_dataset)
    example_batch = train_dataset.sample(())

    agent_class = agents[config['agent_name']]
    agent = agent_class.create(FLAGS.seed, example_batch['observations'], example_batch['actions'], config)

    # Lấy model từ checkpoint (nếu chạy Phase 2)
    if FLAGS.load_checkpoint:
        print(f"🔄 Đang tải trọng số Agent từ: {FLAGS.load_checkpoint}")
        agent = load_agent(FLAGS.load_checkpoint, agent)
        print("✅ Tải trọng số thành công!")

    # ====================== DISCRIMINATOR ======================
    if FLAGS.use_discriminator:
        from models.discriminator import SuccessDiscriminator
        disc_model = SuccessDiscriminator()
        disc_rng = jax.random.PRNGKey(FLAGS.seed + 999)
        disc_params = disc_model.init(disc_rng, 
                                      example_batch['observations'][None], 
                                      example_batch['actions'][None])['params']
        disc_tx = optax.adam(learning_rate=3e-4)
        disc_opt_state = disc_tx.init(disc_params)

        @jax.jit
        # --- SỬA LỖI 1: Thêm rng_key vào tham số ---
        def update_discriminator_step(params, opt_state, batch_obs, batch_acts, batch_labels, rng_key):
            def disc_loss_fn(p):
                # --- SỬA LỖI 2: deterministic=False và cấp key cho Dropout ---
                pred = disc_model.apply({'params': p}, batch_obs, batch_acts, 
                                        deterministic=False, rngs={'dropout': rng_key})
                return -jnp.mean(batch_labels * jnp.log(pred + 1e-8) + 
                               (1 - batch_labels) * jnp.log(1 - pred + 1e-8))
            grads = jax.grad(disc_loss_fn)(params)
            updates, new_opt_state = disc_tx.update(grads, opt_state)
            new_params = optax.apply_updates(params, updates)
            return new_params, new_opt_state

        print(f"✅ Discriminator ENABLED | Init beta = {FLAGS.disc_beta} | Decay ratio = {FLAGS.disc_decay_ratio}")

    # ====================== LOGGING ======================
    prefixes = ["eval", "env"]
    if FLAGS.offline_steps > 0: prefixes.append("offline_agent")
    if FLAGS.online_steps > 0: prefixes.append("online_agent")

    logger = LoggingHelper(
        csv_loggers={prefix: CsvLogger(os.path.join(FLAGS.save_dir, f"{prefix}.csv")) for prefix in prefixes},
        wandb_logger=wandb,
    )

    # ====================== OFFLINE RL ======================
    for i in tqdm.tqdm(range(1, FLAGS.offline_steps + 1), desc="Offline"):
        log_step += 1
        if FLAGS.ogbench_dataset_dir is not None and FLAGS.dataset_replace_interval != 0 and i % FLAGS.dataset_replace_interval == 0:
            dataset_idx = (dataset_idx + 1) % len(dataset_paths)
            train_dataset, val_dataset = make_ogbench_env_and_datasets(
                FLAGS.env_name, dataset_path=dataset_paths[dataset_idx],
                compact_dataset=False, dataset_only=True, cur_env=env)
            train_dataset = process_train_dataset(train_dataset)

        batch = train_dataset.sample_sequence(config['batch_size'], sequence_length=FLAGS.horizon_length, discount=discount)
        agent, offline_info = agent.update(batch)

        if i % FLAGS.log_interval == 0:
            logger.log(offline_info, "offline_agent", step=log_step)

        if i == FLAGS.offline_steps - 1 or (FLAGS.eval_interval != 0 and i % FLAGS.eval_interval == 0):
            eval_info, _, _ = evaluate(
                agent=agent, env=eval_env, action_dim=example_batch["actions"].shape[-1],
                num_eval_episodes=FLAGS.eval_episodes, num_video_episodes=FLAGS.video_episodes,
                video_frame_skip=FLAGS.video_frame_skip)
            logger.log(eval_info, "eval", step=log_step)

    # ====================== LƯU OFFLINE CHECKPOINT ======================
    # --- THÊM MỚI: Tính năng lưu model offline ---
    if FLAGS.save_offline and FLAGS.offline_steps > 0:
        checkpoint_dir = os.path.join(FLAGS.save_dir, "offline_checkpoint")
        os.makedirs(checkpoint_dir, exist_ok=True)
        save_agent(agent, checkpoint_dir)
        print(f"💾 Đã lưu Offline Agent thành công tại: {checkpoint_dir}")

    # ====================== ONLINE RL ======================
    replay_buffer = ReplayBuffer.create_from_initial_dataset(
        dict(train_dataset), size=max(FLAGS.buffer_size, train_dataset.size + 1))

    ob, _ = env.reset()
    action_queue = []
    action_dim = example_batch["actions"].shape[-1]
    current_episode = []
    reward_normalizer = RunningMeanStd()

    for i in tqdm.tqdm(range(1, FLAGS.online_steps + 1), desc="Online"):
        # Tránh lỗi log_step đè lên biểu đồ cũ khi load checkpoint chạy tiếp
        if FLAGS.offline_steps == 0 and i == 1:
            log_step = 1000000 
            
        log_step += 1
        online_rng, key = jax.random.split(online_rng)

        if len(action_queue) == 0:
            action = agent.sample_actions(observations=ob, rng=key)
            action_chunk = np.array(action).reshape(-1, action_dim)
            for a in action_chunk:
                action_queue.append(a)
        action = action_queue.pop(0)

        next_ob, int_reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        transition = dict(
            observations=ob,
            actions=action,
            rewards=float(int_reward),
            terminals=float(done),
            masks=1.0 - terminated,
            next_observations=next_ob,
            is_success=0.0,
        )
        current_episode.append(transition)

        env_info = {k: v for k, v in info.items() if k.startswith("distance")}
        if 'success' in info:
            env_info['success'] = float(info['success'])
        logger.log(env_info, "env", step=log_step)

        if 'antmaze' in FLAGS.env_name and any(x in FLAGS.env_name for x in ['diverse', 'play', 'umaze']):
            int_reward -= 1.0
        elif is_robomimic_env(FLAGS.env_name):
            int_reward -= 1.0
        if FLAGS.sparse:
            int_reward = -1.0 if int_reward != 0.0 else 0.0

        if done:
            is_success = float(info.get('success', False))
            obs_batch = np.stack([t['observations'] for t in current_episode])
            acts_batch = np.stack([t['actions'] for t in current_episode])

            if FLAGS.use_discriminator:
                # --- SỬA LỖI 3: Thêm deterministic=True khi đánh giá ---
                d_probs = np.array(disc_model.apply({'params': disc_params}, obs_batch, acts_batch, deterministic=True))
                
                r_discs_raw = -np.log(1.0 - d_probs + 1e-8).flatten()

                reward_normalizer.update(r_discs_raw)
                r_discs_norm = (r_discs_raw - reward_normalizer.mean) / (np.sqrt(reward_normalizer.var) + 1e-8)

                # --- SỬA LỖI 4: Sử dụng disc_decay_ratio ---
                decay_threshold = FLAGS.online_steps * FLAGS.disc_decay_ratio
                if decay_threshold > 0:
                    beta_t = max(0.0, FLAGS.disc_beta * (1.0 - i / decay_threshold))
                else:
                    beta_t = 0.0

                for idx, t in enumerate(current_episode):
                    t['is_success'] = is_success
                    t['rewards'] = float(t['rewards']) + float(beta_t * r_discs_norm[idx])
                    replay_buffer.add_transition(t)
                
                logger.log({"disc/beta": beta_t, "disc/r_norm_mean": np.mean(r_discs_norm)}, "online_agent", step=log_step)
            else:
                for t in current_episode:
                    t['is_success'] = is_success
                    replay_buffer.add_transition(t)

            current_episode = []
            ob, _ = env.reset()
            action_queue = []
        else:
            ob = next_ob

        if FLAGS.use_discriminator and i >= FLAGS.start_training and i % FLAGS.disc_update_interval == 0:
            batch = replay_buffer.sample_with_success_label(256)
            labels = batch['is_success'][:, None]
            
            # --- SỬA LỖI 5: Tách và truyền rng_key cho Dropout lúc train Discriminator ---
            online_rng, dropout_key = jax.random.split(online_rng)
            disc_params, disc_opt_state = update_discriminator_step(
                disc_params, disc_opt_state, batch['observations'], batch['actions'], labels, dropout_key)

        if i >= FLAGS.start_training:
            batch = replay_buffer.sample_sequence(config['batch_size'] * FLAGS.utd_ratio, 
                        sequence_length=FLAGS.horizon_length, discount=discount)
            batch = jax.tree.map(lambda x: x.reshape((FLAGS.utd_ratio, config["batch_size"]) + x.shape[1:]), batch)
            agent, update_info = agent.batch_update(batch)

            if i % FLAGS.log_interval == 0:
                logger.log(update_info, "online_agent", step=log_step)

        if FLAGS.eval_interval != 0 and i % FLAGS.eval_interval == 0:
            eval_info, _, _ = evaluate(
                agent=agent, env=eval_env, action_dim=action_dim,
                num_eval_episodes=FLAGS.eval_episodes,
                num_video_episodes=FLAGS.video_episodes,
                video_frame_skip=FLAGS.video_frame_skip)
            logger.log(eval_info, "eval", step=log_step)

    for key, csv_logger in logger.csv_loggers.items():
        csv_logger.close()

    with open(os.path.join(FLAGS.save_dir, 'token.tk'), 'w') as f:
        f.write(run.url)

    print("✅ Training completed successfully!")

if __name__ == '__main__':
    app.run(main)