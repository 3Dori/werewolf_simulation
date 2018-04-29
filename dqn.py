import tensorflow as tf
import numpy as np

AVAIABLE_MAX_VAL = 2 ** 12

class DQN(object):
	def __init__(
		self,
		name,
		n_actions,
		n_features,
		learning_rate=0.01,
		reward_delay=0.9,
		n_l1=10,
		e_greedy=0.9,
		replace_target_iter=300,
		memory_size=500,
		batch_size=32,
		e_greedy_increment=None,
	):
		self.name = name
		self.n_actions = n_actions
		self.n_features = n_features
		self.learning_rate = learning_rate
		self.gamma = reward_delay		
		self.n_l1 = n_l1
		self.epsilon_max = e_greedy
		self.replace_target_iter = replace_target_iter
		self.memory_size = memory_size
		self.memory_counter = 0
		self.batch_size = batch_size
		self.epsilon_increment = e_greedy_increment
		self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max

		self.learn_step_counter = 0
		self.memory = np.zeros((self.memory_size, n_features*2+2))

		self._build_net()

		t_params = tf.get_collection(self.name + "_target_net_params")
		e_params = tf.get_collection(self.name + "_eval_net_params")
		self.replace_target_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

		self.sess = tf.Session()
		self.sess.run(tf.global_variables_initializer())
		self.cost_history = []

	def store_transition(self, s, a, r, s_):
		transition = np.hstack((s, a, r, s_))
		index = self.memory_counter % self.memory_size
		self.memory[index, :] = transition
		self.memory_counter += 1
		if self.memory_counter > 200 and self.memory_counter % 5 == 0:
			self.learn()

	def choose_action(self, s, available_actions=None):
		s = np.array(s)
		s = s[np.newaxis, :]

		if np.random.uniform() < self.epsilon:
			actions_value = self.sess.run(self.q_eval, feed_dict={self.s: s})
			if available_actions is not None:
				actions_value += available_actions * AVAIABLE_MAX_VAL
			action = np.argmax(actions_value)
		else:
			if available_actions is None:
				action = np.random.randint(0, self.n_actions)
			else:
				action = np.random.choice(np.nonzero(available_actions)[0])
		return action

	def learn(self):
		if self.learn_step_counter % self.replace_target_iter == 0:
			self.sess.run(self.replace_target_op)

		if self.memory_counter > self.memory_size:
			sample_index = np.random.choice(self.memory_size, size=self.batch_size)
		else:
			sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
		batch_memory = self.memory[sample_index, :]

		_, cost = self.sess.run(
			[self._train_op, self.loss],
			feed_dict={
				self.s: batch_memory[:, :self.n_features],
				self.s_: batch_memory[:, -self.n_features:],
				self.a: batch_memory[:, self.n_features],
				self.r: batch_memory[:, self.n_features+1]
			}
		)

		self.cost_history.append(cost)

		self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max

	def _build_net(self):
		self.s = tf.placeholder(tf.float32, [None, self.n_features], name="s")
		self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name="s_")
		self.r = tf.placeholder(tf.float32, [None, ], name="r")
		self.a = tf.placeholder(tf.int32, [None, ], name="a")

		with tf.variable_scope(self.name + "_eval_net"):
			c_names = [self.name + "_eval_net_params", tf.GraphKeys.GLOBAL_VARIABLES]
			w_initializer = tf.random_normal_initializer(0.0, 0.3)
			b_initializer = tf.constant_initializer(0.1)

			with tf.variable_scope("l1"):
				w1 = tf.get_variable('w1', [self.n_features, self.n_l1],
					initializer=w_initializer, collections=c_names)
				b1 = tf.get_variable('b1', [1, self.n_l1],
					initializer=b_initializer, collections=c_names)
				l1 = tf.nn.relu(tf.matmul(self.s, w1) + b1)

			with tf.variable_scope("l2"):
				w2 = tf.get_variable('w2', [self.n_l1, self.n_actions],
					initializer=w_initializer, collections=c_names)
				b2 = tf.get_variable('b2', [1, self.n_actions],
					initializer=b_initializer, collections=c_names)
				self.q_eval = tf.matmul(l1, w2) + b2

		with tf.variable_scope(self.name + "_target_net"):
			c_names = [self.name + "_target_net_params", tf.GraphKeys.GLOBAL_VARIABLES]

			with tf.variable_scope("l1"):
				w1 = tf.get_variable('w1', [self.n_features, self.n_l1],
					initializer=w_initializer, collections=c_names)
				b1 = tf.get_variable('b1', [1, self.n_l1],
					initializer=b_initializer, collections=c_names)
				l1 = tf.nn.relu(tf.matmul(self.s_, w1) + b1)

			with tf.variable_scope("l2"):
				w2 = tf.get_variable('w2', [self.n_l1, self.n_actions],
					initializer=w_initializer, collections=c_names)
				b2 = tf.get_variable('b2', [1, self.n_actions],
					initializer=b_initializer, collections=c_names)
				self.q_next = tf.matmul(l1, w2) + b2

		with tf.variable_scope(self.name + "_q_target"):
			q_target = self.r + self.gamma * tf.reduce_max(self.q_next, axis=1, name="Qmax_s_")
			self.q_target = tf.stop_gradient(q_target)
		with tf.variable_scope(self.name + "_q_eval"):
			a_indices = tf.stack([tf.range(tf.shape(self.a)[0], dtype=tf.int32), self.a], axis=1)
			self.q_eval_wrt_a = tf.gather_nd(params=self.q_eval, indices=a_indices)
		with tf.variable_scope(self.name + "_loss"):
			self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval_wrt_a, name="TD_error"))
		with tf.variable_scope(self.name + "_train"):
			self._train_op = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss)
