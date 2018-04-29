from dqn import DQN

class GameNetwork(object):
	def __init__(
		self,
		player_num=9,
		use_vote_feature=False,
		use_three_atitutes=True,
		has_captain=False
	):
		self.use_vote_feature = use_vote_feature
		self.player_num = player_num
		self.use_three_atitutes = use_three_atitutes
		self.has_captain = has_captain

		n_basic_features = player_num + player_num * player_num * (2 if use_vote_feature else 1)
		self.n_villager_features = n_basic_features
		self.n_werewolf_features = n_basic_features + player_num
		self.n_seer_features = n_basic_features + player_num
		self.n_witch_features = n_basic_features + player_num
		self.n_hunter_features = n_basic_features

		self.n_utter_actions = player_num * (3 if use_three_atitutes else 2)
		self.n_vote_actions = player_num

		self.n_werewolf_kill_actions = player_num
		self.n_seer_identify_actions = player_num
		self.n_witch_save_actions = 2
		self.n_witch_kill_actions = player_num + 1
		self.n_hunter_hunt_actions = player_num

		# Utter networks
		self.villager_utter_net = DQN("villager_utter", self.n_utter_actions, self.n_villager_features)
		self.werewolf_utter_net = DQN("werewolf_utter", self.n_utter_actions, self.n_werewolf_features)
		self.seer_utter_net = DQN("seer_utter", self.n_utter_actions, self.n_seer_features)
		self.witch_utter_net = DQN("witch_utter", self.n_utter_actions, self.n_witch_features)
		self.hunter_utter_net = DQN("hunter_utter", self.n_utter_actions, self.n_hunter_features)

		# Vote networks
		self.villager_vote_net = DQN("villager_vote", self.n_vote_actions, self.n_villager_features)
		self.werewolf_vote_net = DQN("werewolf_vote", self.n_vote_actions, self.n_werewolf_features)
		self.seer_vote_net = DQN("seer_vote", self.n_vote_actions, self.n_seer_features)
		self.witch_vote_net = DQN("witch_vote", self.n_vote_actions, self.n_witch_features)
		self.hunter_vote_net = DQN("hunter_vote", self.n_vote_actions, self.n_hunter_features)

		# Special networks
		self.werewolf_kill_net = DQN("werewolf_kill", self.n_werewolf_kill_actions, self.n_werewolf_features)
		self.seer_identify_net = DQN("seer_identify", self.n_seer_identify_actions, self.n_seer_features)
		self.witch_save_net = DQN("witch_save", self.n_witch_save_actions, self.n_witch_features)
		self.witch_kill_net = DQN("witch_kill", self.n_witch_kill_actions, self.n_witch_features)
		self.hunter_hunt_net = DQN("hunter_hunt", self.n_hunter_hunt_actions, self.n_hunter_features)