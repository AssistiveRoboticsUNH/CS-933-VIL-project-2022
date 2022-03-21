class ReplayBuffer():
    def __init__(self, env, buffer_capacity=BUFFER_CAPACITY, batch_size=BATCH_SIZE, min_size_buffer=MIN_SIZE_BUFFER):
        self.buffer_capacity = buffer_capacity
        self.batch_size = batch_size
        self.min_size_buffer = min_size_buffer
        self.buffer_counter = 0
        self.n_games = 0
        self.states = np.zeros((self.buffer_capacity, env.observation_space.shape[0]))
        self.actions = np.zeros((self.buffer_capacity, env.action_space.shape[0]))
        self.rewards = np.zeros((self.buffer_capacity))
        self.next_states = np.zeros((self.buffer_capacity, env.observation_space.shape[0]))
        self.dones = np.zeros((self.buffer_capacity), dtype=bool)
        
        
    def __len__(self):
        return self.buffer_counter


    def add_record(self, state, action, reward, next_state, done):
        # Set index to zero if counter = buffer_capacity and start again (1 % 100 = 1 and 101 % 100 = 1) so we substitute the older entries
        index = self.buffer_counter % self.buffer_capacity
        self.states[index] = state
        self.actions[index] = action
        self.rewards[index] = reward
        self.next_states[index] = next_state
        self.dones[index] = done
        
        # Update the counter when record something
        self.buffer_counter += 1
    

    def check_buffer_size(self):
        return self.buffer_counter >= self.batch_size and self.buffer_counter >= self.min_size_buffer
    

    def update_n_games(self):
        self.n_games += 1
    

    def get_minibatch(self):
        # If the counter is less than the capacity we don't want to take zeros records, 
        # if the cunter is higher we don't access the record using the counter 
        # because older records are deleted to make space for new one
        buffer_range = min(self.buffer_counter, self.buffer_capacity)
        
        batch_index = np.random.choice(buffer_range, self.batch_size, replace=False)
        # Convert to tensors
        state = self.states[batch_index]
        action = self.actions[batch_index]
        reward = self.rewards[batch_index]
        next_state = self.next_states[batch_index]
        done = self.dones[batch_index]
        return state, action, reward, next_state, done
    

    def save(self, folder_name):
        """
        Save the replay buffer
        """
        if not os.path.isdir(folder_name):
            os.mkdir(folder_name)
        np.save(folder_name + '/states.npy', self.states)
        np.save(folder_name + '/actions.npy', self.actions)
        np.save(folder_name + '/rewards.npy', self.rewards)
        np.save(folder_name + '/next_states.npy', self.next_states)
        np.save(folder_name + '/dones.npy', self.dones)
        
        dict_info = {"buffer_counter": self.buffer_counter, "n_games": self.n_games}
        
        with open(folder_name + '/dict_info.json', 'w') as f:
            json.dump(dict_info, f)


    def load(self, folder_name):
        """
        Load the replay buffer
        """
        self.states = np.load(folder_name + '/states.npy')
        self.actions = np.load(folder_name + '/actions.npy')
        self.rewards = np.load(folder_name + '/rewards.npy')
        self.next_states = np.load(folder_name + '/next_states.npy')
        self.dones = np.load(folder_name + '/dones.npy')
        
        with open(folder_name + '/dict_info.json', 'r') as f:
            dict_info = json.load(f)
        self.buffer_counter = dict_info["buffer_counter"]
        self.n_games = dict_info["n_games"]

class Actor(tf.keras.Model):
    def __init__(self, name, actions_dim, upper_bound, hidden_0=CRITIC_HIDDEN_0, hidden_1=CRITIC_HIDDEN_1, init_minval=INIT_MINVAL, init_maxval=INIT_MAXVAL):
        super(Actor, self).__init__()
        self.hidden_0 = hidden_0
        self.hidden_1 = hidden_1
        self.actions_dim = actions_dim
        self.init_minval = init_minval
        self.init_maxval = init_maxval
        self.upper_bound = upper_bound
        
        self.net_name = name
        self.dense_0 = Dense(self.hidden_0, activation='relu')
        self.dense_1 = Dense(self.hidden_1, activation='relu')
        self.policy = Dense(self.actions_dim, kernel_initializer=random_uniform(
                            minval=self.init_minval, maxval=self.init_maxval),
                            activation='tanh')


    def call(self, state):
        x = self.dense_0(state)
        policy = self.dense_1(x)
        policy = self.policy(policy)
        return policy * self.upper_bound

class Critic(tf.keras.Model):
    def __init__(self, name, hidden_0=CRITIC_HIDDEN_0, hidden_1=CRITIC_HIDDEN_1):
        super(Critic, self).__init__()
        
        self.hidden_0 = hidden_0
        self.hidden_1 = hidden_1
        self.net_name = name
        self.dense_0 = Dense(self.hidden_0, activation='relu')
        self.dense_1 = Dense(self.hidden_1, activation='relu')
        self.q_value = Dense(1, activation=None)


    def call(self, state, action):
        state_action_value = self.dense_0(tf.concat([state, action], axis=1))
        state_action_value = self.dense_1(state_action_value)
        q_value = self.q_value(state_action_value)
        return q_value

class Agent:
    def __init__(self, env, actor_lr=ACTOR_LR, critic_lr=CRITIC_LR, gamma=GAMMA, max_size=BUFFER_CAPACITY, tau=TAU, path_save=PATH_SAVE, path_load=PATH_LOAD):
        
        self.gamma = gamma
        self.tau = tau
        self.replay_buffer = ReplayBuffer(env, max_size)
        self.actions_dim = env.action_space.shape[0]
        self.upper_bound = env.action_space.high[0]
        self.lower_bound = env.action_space.low[0]
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.path_save = path_save
        self.path_load = path_load
        
        self.actor = Actor(name='actor', actions_dim=self.actions_dim,
                           upper_bound=self.upper_bound)
        self.critic = Critic(name='critic')
        self.target_actor = Actor(name='target_actor',
                                  actions_dim=self.actions_dim,
                                  upper_bound=self.upper_bound)
        self.target_critic = Critic(name='target_critic')
        self.actor.compile(optimizer=opt.Adam(learning_rate=actor_lr))
        self.critic.compile(optimizer=opt.Adam(learning_rate=critic_lr))
        self.target_actor.compile(optimizer=opt.Adam(learning_rate=actor_lr))
        self.target_critic.compile(optimizer=opt.Adam(learning_rate=critic_lr))
        actor_weights = self.actor.get_weights()
        critic_weights = self.critic.get_weights()
        
        self.target_actor.set_weights(actor_weights)
        self.target_critic.set_weights(critic_weights)
        
        self.noise = np.zeros(self.actions_dim)

    def update_target_networks(self, tau):
        actor_weights = self.actor.weights
        target_actor_weights = self.target_actor.weights
        for index in range(len(actor_weights)):
            target_actor_weights[index] = tau * actor_weights[index] +
                                        (1 - tau) * target_actor_weights[index]
            self.target_actor.set_weights(target_actor_weights)
        
        critic_weights = self.critic.weights
        target_critic_weights = self.target_critic.weights
    
        for index in range(len(critic_weights)):
            target_critic_weights[index] = tau * critic_weights[index] +
                                        (1 - tau) * target_critic_weights[index]
            self.target_critic.set_weights(target_critic_weights)

    def add_to_replay_buffer(self, state, action, reward, new_state, done):
        self.replay_buffer.add_record(state, action, reward, new_state, done)
        

    def save(self):
        date_now = time.strftime("%Y%m%d%H%M")
        if not os.path.isdir(f"{self.path_save}/save_agent_{ENV_NAME.lower()}_{date_now}"):
            os.makedirs(f"{self.path_save}/save_agent_{ENV_NAME.lower()}_{date_now}")
        self.actor.save_weights(f"{self.path_save}/save_agent_{ENV_NAME.lower()}_{date_now}/{self.actor.net_name}.h5")
        self.critic_0.save_weights(f"{self.path_save}/save_agent_{ENV_NAME.lower()}_{date_now}/{self.critic_0.net_name}.h5")
        self.critic_1.save_weights(f"{self.path_save}/save_agent_{ENV_NAME.lower()}_{date_now}/{self.critic_1.net_name}.h5")
        self.critic_value.save_weights(f"{self.path_save}/save_agent_{ENV_NAME.lower()}_{date_now}/{self.critic_value.net_name}.h5")
        self.critic_target_value.save_weights(f"{self.path_save}/save_agent_{ENV_NAME.lower()}_{date_now}/{self.critic_target_value.net_name}.h5")
        
        self.replay_buffer.save(f"{self.path_save}/save_agent_{ENV_NAME.lower()}_{date_now}")


    def load(self):
        self.actor.load_weights(f"{self.path_load}/{self.actor.net_name}.h5")
        self.critic_0.load_weights(f"{self.path_load}/{self.critic_0.net_name}.h5")
        self.critic_1.load_weights(f"{self.path_load}/{self.critic_1.net_name}.h5")
        self.critic_value.load_weights(f"{self.path_load}/{self.critic_value.net_name}.h5")
        self.critic_target_value.load_weights(f"{self.path_load}/{self.critic_target_value.net_name}.h5")
        
        self.replay_buffer.load(f"{self.path_load}")


    def get_action(self, observation):
        state = tf.convert_to_tensor([observation])
        actions, _ = self.actor.get_action_log_probs(state, reparameterization_trick=False)
        return actions[0]

    def _ornstein_uhlenbeck_process(self, x, theta=THETA, mu=0, dt=DT, std=0.2):
        """
        Ornstein–Uhlenbeck process
        """
        return x + theta * (mu-x) * dt + std * np.sqrt(dt) *
                                np.random.normal(size=self.actions_dim)


    def get_action(self, observation, noise, evaluation=False):
        state = tf.convert_to_tensor([observation], dtype=tf.float32)
        actions = self.actor(state)
        if not evaluation:
            self.noise = self._ornstein_uhlenbeck_process(noise)
            actions += self.noise
            actions = tf.clip_by_value(actions, self.lower_bound, self.upper_bound)
            return actions[0]

    def learn(self):
        if self.replay_buffer.check_buffer_size() == False:
            return

        state, action, reward, new_state, done =
                                self.replay_buffer.get_minibatch()
        states = tf.convert_to_tensor(state, dtype=tf.float32)
        new_states = tf.convert_to_tensor(new_state, dtype=tf.float32)
        rewards = tf.convert_to_tensor(reward, dtype=tf.float32)
        actions = tf.convert_to_tensor(action, dtype=tf.float32)

        with tf.GradientTape() as tape:
            target_actions = self.target_actor(new_states)
            target_critic_values = tf.squeeze(self.target_critic(new_states,
                                                        target_actions), 1)
            critic_value = tf.squeeze(self.critic(states, actions), 1)
            target = reward + self.gamma * target_critic_values * (1-done)
            critic_loss = tf.keras.losses.MSE(target, critic_value)
            critic_gradient = tape.gradient(critic_loss,
                                            self.critic.trainable_variables)

        self.critic.optimizer.apply_gradients(zip(critic_gradient,
                                                self.critic.trainable_variables))

        with tf.GradientTape() as tape:
            policy_actions = self.actor(states)
            actor_loss = -self.critic(states, policy_actions)
            actor_loss = tf.math.reduce_mean(actor_loss)
            actor_gradient = tape.gradient(actor_loss, 
                                    self.actor.trainable_variables)

        self.actor.optimizer.apply_gradients(zip(actor_gradient,
                                                self.actor.trainable_variables))
        self.update_target_networks(self.tau)

config = dict(
  learning_rate_actor=ACTOR_LR,
  learning_rate_critic=ACTOR_LR,
  batch_size=BATCH_SIZE,
  architecture="DDPG",
  infra="Manjaro",
  env=ENV_NAME
)

wandb.init(
  project=f"tensorflow2_ddpg_{ENV_NAME.lower()}",
  tags=["DDPG", "FCL", "RL"],
  config=config,
)

env = gym.make(ENV_NAME)
agent = Agent(env)
scores = []
evaluation = True

if PATH_LOAD is not None:
    print("loading weights")
    observation = env.reset()
    action = agent.actor(observation[None, :])
    agent.target_actor(observation[None, :])
    agent.critic(observation[None, :], action)
    agent.target_critic(observation[None, :], action)
    agent.load()
    print(agent.replay_buffer.buffer_counter)
    print(agent.replay_buffer.n_games)
    print(agent.noise)

for _ in tqdm(range(MAX_GAMES)):
    start_time = time.time()
    states = env.reset()
    done = False
    score = 0
    while not done:
        action = agent.get_action(states, evaluation)
        new_states, reward, done, info = env.step(action)
        score += reward
        agent.add_to_replay_buffer(states, action, reward, new_states, done)
        agent.learn()
        states = new_states
        
    agent.replay_buffer.update_n_games()
    
    scores.append(score)
        wandb.log({'Game number': agent.replay_buffer.n_games,
                   '# Episodes': agent.replay_buffer.buffer_counter, 
                   "Average reward": round(np.mean(scores[-10:]), 2),
                    "Time taken": round(time.time() - start_time, 2)})
        if (_ + 1) % EVALUATION_FREQUENCY == 0:
        evaluation = True
        states = env.reset()
        done = False
        score = 0
        while not done:
            action = agent.get_action(states, evaluation)
            new_states, reward, done, info = env.step(action)
            score += reward
            states = new_states
        wandb.log({'Game number': agent.replay_buffer.n_games, 
                   '# Episodes': agent.replay_buffer.buffer_counter, 
                   'Evaluation score': score})
        evaluation = False
     
    if (_ + 1) % SAVE_FREQUENCY == 0:
        print("saving...")
        agent.save()
        print("saved")


