from game_network import GameNetwork

import numpy as np

import random
from collections import Counter
from enum import Enum

class Role(Enum):
	UNKNOWN = 0
	VILLAGER = 1
	WEREWOLF = 2
	SEER = 3
	WITCH = 4
	HUNTER = 5


class GameState(Enum):
	WEREWOLF_WIN = 0
	NONEWEREWOLF_WIN = 1
	NOT_ENDED = 2

	
class Player(object):
	def __init__(self, id, use_vote_feature=False):
		self.id = id
		self.score = 0
		self.update_transitions_count = 0
		self.use_vote_feature = use_vote_feature

	def assign_role(self, role, player_num, alives, utterances):
		self.role = role
		self.player_num = player_num
		self.guess = [0] * player_num    # 0: unknown, 1: villager camp, -1: werewolf camp
		self.guess[self.id] = -1 if role == Role.WEREWOLF else 1
		self.is_alive = True
		self.is_captain = False
		self.alives = alives
		self.utterances = utterances
		self.witch_saved = False
		self.witch_killed = False
		self.hunt = -1
		self.transitions = []

	def get_id(self):
		return self.id

	def get_role(self):
		return self.role

	def get_transitions(self):
		return self.transitions

	def set_death(self):
		self.is_alive = False

	def identify(self, player_id, role):
		self.guess[player_id] = -1 if role == Role.WEREWOLF else 1

	def _to_game_id(self, player_id, network_id):
		if network_id == 0:
			return player_id
		elif network_id <= player_id:
			return network_id - 1
		else:
			return network_id

	def _to_network_id(self, player_id, game_id):
		# Transform game id to network id
		if player_id == game_id:
			return 0
		elif player_id < game_id:
			return game_id
		else:
			return game_id + 1

	def generate_state(self):
		utterance_vec = [0] * (self.player_num * self.player_num)
		for i, utterance in enumerate(self.utterances):
			target = utterance[0]
			attitude = utterance[1]
			if target != -1:
				index = self._to_network_id(self.id, target) + i * self.player_num
				utterance_vec[index] = attitude

		if self.use_vote_feature:
			raise NotImplementedError("Use vote feature not implememted")

		alive_vec = [1] * self.player_num
		for i in range(self.player_num):
			index = self._to_network_id(self.id, i)
			alive_vec[index] = 1 if self.alives[i] else 0

		if self.role == Role.WEREWOLF or self.role == Role.SEER:
			identity_vec = [0] * self.player_num
			for i in range(self.player_num):
				index = self._to_network_id(self.id, i)
				identity_vec[index] = self.guess[i]
		elif self.role == Role.WITCH:
			identity_vec = [0] * self.player_num
			if self.next_werewolf_kill != -1:
				index = self._to_network_id(self.id, self.next_werewolf_kill)
				identity_vec[index] = 1
		else:
			identity_vec = []

		return alive_vec + identity_vec + utterance_vec

	def update_transitions(self, game_state):
		trasitions = self.transitions
		next_state = self.generate_state()

		if game_state == GameState.NOT_ENDED:
			reward = 0
		elif game_state == GameState.WEREWOLF_WIN:
			reward = 1 if self.role == Role.WEREWOLF else -1
		elif game_state == GameState.NONEWEREWOLF_WIN:
			reward = -1 if self.role == Role.WEREWOLF else 1
		else:
			raise RuntimeError("Cannot recoginize game state {0}".format(game_state))

		for transition in self.transitions:
			network = transition['network']
			network.store_transition(transition['s'], transition['a'], reward, next_state)

	def clear_transitions(self):
		self.transitions.clear()

	def get_available_actions(self, action):
		# Restrict actions, for example, cannot kill or vote a dead played
		if self.role == Role.WITCH and action == "kill":
			available_actions = np.array([1] * (self.player_num + 1))
		else:
			available_actions = np.array([1] * self.player_num)
		same_camp = -1 if self.role == Role.WEREWOLF else 1
		for i in range(self.player_num):
			index = self._to_network_id(self.id, i)
			if self.role == Role.WEREWOLF and action == "kill":
				availability = self.alives[i] and self.guess[i] != -1
			elif self.role == Role.SEER and action == "identify":
				availability = self.alives[i] and self.guess[i] == 0
			elif self.role == Role.WITCH and action == "kill":
				availability = self.alives[i] and self.guess[i] != 1
			elif self.role == Role.HUNTER and action == "hunt":
				availability = self.alives[i] and self.guess[i] != 1
			elif action == "vote":
				availability = self.alives[i] and self.guess[i] != same_camp
			else:
				raise NotImplementedError("Cannot recoginize action {0}".format(action))
			available_actions[index] = availability
		return available_actions

	def get_player_to_identify(self):
		# return -1
		# Test code: always identify first unidentified alive player
		# candidates = []
		# for i, alive in enumerate(self.alives):
		# 	if alive and i != self.id and self.guess[i] == 0:
		# 		candidates.append(i)
		# return random.choice(candidates)
		state = self.generate_state()
		network = game_network.seer_identify_net
		player_to_identify = network.choose_action(state, self.get_available_actions("identify"))
		self.transitions.append({
			's': state,
			'a': player_to_identify,
			'network': network
		})
		return self._to_game_id(self.id, player_to_identify)

	def get_player_to_kill(self):
		assert self.role == Role.WEREWOLF
		# candidates = []
		# for i, alive in enumerate(self.alives):
		# 	if alive and self.guess[i] != 1:
		# 		candidates.append(i)
		# return random.choice(candidates)
		state = self.generate_state()
		network = game_network.werewolf_kill_net
		player_to_kill = network.choose_action(state, self.get_available_actions("kill"))
		self.transitions.append({
			's': state,
			'a': player_to_kill,
			'network': network
		})
		return self._to_game_id(self.id, player_to_kill)

	def determine_to_save(self, next_werewolf_kill):
		assert self.role == Role.WITCH
		# Test code: 1 / 5 chance to save
		# return random.random() > 0.8
		self.next_werewolf_kill = next_werewolf_kill
		state = self.generate_state()
		network = game_network.witch_save_net
		available_actions = np.array([1, 1])
		if self.witch_saved or next_werewolf_kill == -1:
			available_actions[0] = 0
		saved = network.choose_action(state, available_actions)
		self.transitions.append({
			's': state,
			'a': saved,
			'network': network
		})
		return [True, False][saved]

	def get_player_to_witch_kill(self):
		assert self.role == Role.WITCH
		# candidates = [-1]
		# for i, alive in enumerate(self.alives):
		# 	if alive and i != self.id:
		# 		candidates.append(i)
		# return random.choice(candidates)
		state = self.generate_state()
		network = game_network.witch_kill_net
		player_to_witch_kill = network.choose_action(state, self.get_available_actions("kill"))
		self.transitions.append({
			's': state,
			'a': player_to_witch_kill,
			'network': network
		})
		self.next_werewolf_kill = -1
		if player_to_witch_kill == self.player_num:
			return -1
		else:
			return self._to_game_id(self.id, player_to_witch_kill)

	def get_player_to_hunt(self):
		assert self.role == Role.HUNTER
		# candidates = []
		# for i, alive in enumerate(self.alives):
		# 	if alive and i != self.id:
		# 		candidates.append(i)
		# return random.choice(candidates)
		state = self.generate_state()
		network = game_network.hunter_hunt_net
		player_to_hunt = network.choose_action(state, self.get_available_actions("hunt"))
		self.transitions.append({
			's': state,
			'a': player_to_hunt,
			'network': network
		})
		return self._to_game_id(self.id, player_to_hunt)

	def get_player_to_vote(self, vote_from=None):
		# Test code: always vote first alive player
		# candidates = []
		# if vote_from is not None:
		# 	for i in vote_from:
		# 		if self.alives[i] and i != self.id:
		# 			candidates.append(i)
		# else:
		# 	for i, alive in enumerate(self.alives):
		# 		if alive and i != self.id:
		# 			candidates.append(i)
		# return random.choice(candidates)
		state = self.generate_state()
		if self.role == Role.VILLAGER:
			network = game_network.villager_vote_net
		elif self.role == Role.WEREWOLF:
			network = game_network.werewolf_vote_net
		elif self.role == Role.SEER:
			network = game_network.seer_vote_net
		elif self.role == Role.WITCH:
			network = game_network.witch_vote_net
		elif self.role == Role.HUNTER:
			network = game_network.hunter_vote_net

		available_actions = self.get_available_actions("vote")
		if vote_from is not None:
			vote_from_list = np.array([0] * self.player_num)
			for player_id in vote_from:
				vote_from_list[self._to_network_id(self.id, player_id)] = 1
			available_actions *= vote_from_list

		player_to_vote = network.choose_action(state, available_actions)
		self.transitions.append({
			's': state,
			'a': player_to_vote,
			'network': network
		})
		return self._to_game_id(self.id, player_to_vote)

	def get_utterance(self, use_three_attitudes=True):
		assert self.is_alive
		# target = random.randint(0, self.player_num)
		# attitude = random.randint(-1, 1)
		# return target, attitude
		state = self.generate_state()
		if self.role == Role.VILLAGER:
			network = game_network.villager_utter_net
		elif self.role == Role.WEREWOLF:
			network = game_network.werewolf_utter_net
		elif self.role == Role.SEER:
			network = game_network.seer_utter_net
		elif self.role == Role.WITCH:
			network = game_network.witch_utter_net
		elif self.role == Role.HUNTER:
			network = game_network.hunter_utter_net
		else:
			raise NotImplementedError("Cannot recoginize role {0}".format(self.role))
		action = network.choose_action(state)
		# Decode action
		# 0-bad 1-bad 2-bad ... 8-bad 0-neu 1-neu ... 8-neu 0-good 1-good ... 8-good
		target = action % 9
		if use_three_attitudes:
			attitude = target // 9 - 1
		else:
			attitude = -1 if target < 9 else 1
		self.transitions.append({
			's': state,
			'a': action,
			'network': network
		})
		return self._to_game_id(self.id, target), attitude

	def get_player_to_elect(self, vote_from=None):
		raise NotImplementedError("Captain choose not implemented")
		# Test code: always vote first alive player
		candidates = []
		if vote_from is not None:
			for i in vote_from:
				if self.alives[i] and i != self.id:
					candidates.append(i)
		else:
			for i, alive in enumerate(self.alives):
				if alive and i != self.id:
					candidates.append(i)
		return random.choice(candidates)

	def set_witch_saved(self):
		self.witch_saved = True

	def set_witch_killed(self):
		self.witch_killed = True

	def choose_next_captain(self):
		raise NotImplementedError("Captain choose not implemented")
		assert self.is_captain
		self.is_captain = False
		# Test code: always choose first alive player
		candidates = []
		for i, alive in enumerate(self.alives):
			if alive and i != self.id:
				candidates.append(i)
		return random.choice(candidates)

	def __repr__(self):
		return "Player {0}: score {1}".format(self.id, self.score)


class Game(object):
	def __init__(self, roles, display_process=True, use_vote_feature=False, has_captain=True):
		for role in roles:
			if role not in [Role.VILLAGER, Role.WEREWOLF, Role.SEER, Role.WITCH, Role.HUNTER]:
				raise RuntimeError('Unsupported role {0}'.format(role))
		roles_counter = Counter(roles)
		self.player_num = len(roles)
		if roles_counter[Role.WEREWOLF] == 0:
			raise RuntimeError('Should have at least one werewolf')
		if self.player_num - roles_counter[Role.WEREWOLF] == 0:
			raise RuntimeError('Should have at least one villager, witch, seer or hunter')
		if roles_counter[Role.WITCH] > 1:
			raise RuntimeError('Can have at most one witch')
		if roles_counter[Role.HUNTER] > 1:
			raise RuntimeError('Can have at most one hunter')
		if roles_counter[Role.SEER] > 1:
			raise RuntimeError('Can have at most one SEER')

		self.roles = roles
		self.display_process = display_process
		self.use_vote_feature = use_vote_feature
		self.has_captain = has_captain
		self.players = []
		for i in range(self.player_num):
			player = Player(i)
			self.players.append(player)

	def new_game(self):
		if self.display_process:
			print("\n\n New game")
		current_roles = list(self.roles)
		random.shuffle(current_roles)
		self.werewolves = []
		self.nonwerewolves = []
		self.villagers = []
		self.alives = [True] * self.player_num
		self.utterances = [(-1, 0)] * self.player_num
		self.votes = [-1] * self.player_num
		self.captain_id = -1
		for i, role in enumerate(current_roles):
			self.players[i].assign_role(role, self.player_num, self.alives, self.utterances)
			if role == Role.WEREWOLF:
				self.werewolves.append(self.players[i])
			else:
				self.nonwerewolves.append(self.players[i])
			if role == Role.VILLAGER:
				self.villagers.append(self.players[i])
			if role == Role.SEER:
				self.seer = self.players[i]
			if role == Role.WITCH:
				self.witch = self.players[i]
			if role == Role.HUNTER:
				self.hunter = self.players[i]

		# Werewolves identify each other
		for werewolf in self.werewolves:
			for other_werewolf in self.werewolves:
				werewolf.identify(other_werewolf.get_id(), Role.WEREWOLF)
			for nonwerewolf in self.nonwerewolves:
				werewolf.identify(nonwerewolf.get_id(), Role.VILLAGER)

		if self.display_process:
			print(current_roles)
		if self.has_captain:
			self.elect_captain()
		turn = 0

		while True:
			if self.display_process:
				print("Turn {0}".format(turn + 1))
			game_state = self.one_loop()
			if game_state != GameState.NOT_ENDED:
				if self.display_process:
					print(game_state)
				break
			turn += 1

	def batch_update_transitions(self, players, game_state):
		game_state = self.determine_win_condition()
		for player in players:
			if player.is_alive:
				player.update_transitions(game_state)

	def batch_clear_transitions(self, players):
		for player in players:
			player.clear_transitions()

	def elect_captain(self):
		highest_voted_players = self.vote(captain=True)
		if len(highest_voted_players) == 1:
			captain_id = highest_voted_players[0]
		else:
			highest_voted_players = self.vote(highest_voted_players)
			if len(highest_voted_players) == 1:
				captain_id = highest_voted_players[0]
			else:
				captain_id = random.sample(highest_voted_players, 1)
		if self.display_process:
			print("{0} is elected captain".format(captain_id))
		self.players[captain_id].is_captain = True
		self.captain_id = captain_id

	def set_player_killed(self, player_id, voted=True):
		player_to_hunt = -1
		self.players[player_id].set_death()
		self.alives[player_id] = False
		if player_id == self.hunter.get_id() and voted:
			player_to_hunt = self.hunter.get_player_to_hunt()
			if self.display_process:
				print("{0} hunted {1}".format(player_id, player_to_hunt))
			self.players[player_to_hunt].set_death()
			self.alives[player_to_hunt] = False
		if not self.has_captain:
			return
		if player_id == self.captain_id or player_to_hunt == self.captain_id:
			if player_id == self.captain_id:
				next_captain_id = self.players[player_id].choose_next_captain()
			else:
				next_captain_id = self.players[player_to_hunt].choose_next_captain()
			print("{0} is the next captain".format(next_captain_id))
			self.players[next_captain_id].is_captain = True
			self.captain_id = next_captain_id

	def get_player_to_kill(self):
		# candidates = []
		# for player in self.nonwerewolves:
		# 	if player.is_alive:
		# 		candidates.append(player.get_id())
		# return random.choice(candidates)
		votes = [0] * self.player_num
		highest_voted_players = []
		highest_vote = -1
		for werewolf in self.werewolves:
			if werewolf.is_alive:
				votes[werewolf.get_player_to_kill()] += 1
		for i, vote in enumerate(votes):
			if vote > highest_vote:
				highest_voted_players = [i]
				highest_vote = vote
			elif vote == highest_vote:
				highest_voted_players.append(i)
		if len(highest_voted_players) == 1:
			return highest_voted_players[0]
		else:
			return random.choice(highest_voted_players)

	def utter(self):
		for i, player in enumerate(self.players):
			if player.is_alive:
				player.utterances[i] = player.get_utterance()
				if self.display_process:
					print("Player {0} said: Player {1} is {2}".format(
						i, player.utterances[i][0], ["unknown", "good", "bad"][self.utterances[i][1]]
					))

	def vote(self, vote_from=None, captain=True):
		votes = [0] * self.player_num
		highest_voted_players = []
		highest_vote = -1
		for i, player in enumerate(self.players):
			if captain or player.is_alive:
				if captain:
					raise NotImplementedError("Does not support captain election")
					# player_to_vote = player.get_player_to_elect(state, vote_from)
				else:
					player_to_vote = player.get_player_to_vote(vote_from)
				#print("{0} voted {1}".format(player.get_id(), player_to_vote))
				assert self.players[player_to_vote].is_alive
				votes[player_to_vote] += 1.5 if player.is_captain else 1
		for i, vote in enumerate(votes):
			if vote > highest_vote:
				highest_voted_players = [i]
				highest_vote = vote
			elif vote == highest_vote:
				highest_voted_players.append(i)
		return highest_voted_players

	def determine_win_condition(self):
		alive_werewolf_num = 0
		alive_nonwerewolf_num = 0
		for werewolf in self.werewolves:
			if werewolf.is_alive:
				alive_werewolf_num += 1
		for nonwerewolf in self.nonwerewolves:
			if nonwerewolf.is_alive:
				alive_nonwerewolf_num += 1
		if alive_werewolf_num == 0:
			return GameState.NONEWEREWOLF_WIN
		elif alive_nonwerewolf_num <= 1:
			return GameState.WEREWOLF_WIN
		else:
			return GameState.NOT_ENDED

	def one_loop(self):
		# Werewolves kill a player
		player_to_kill = self.get_player_to_kill()
		assert self.players[player_to_kill].is_alive
		if self.display_process:
			print("Werewolves killed {0}".format(player_to_kill))

		# Witch action
		saved = False
		if self.witch.is_alive:
			if not self.witch.witch_saved:
				if self.witch.determine_to_save(player_to_kill):
					if self.display_process:
						print("Witch saved {0}".format(player_to_kill))
					self.witch.set_witch_saved()
				else:
					self.set_player_killed(player_to_kill, False)
				# To handle a special case:
				# Alive players are: witch, villager, werewolf.
				# The werewolf will kill the villager.
				# Below is all cases where the witch is to
				#   save/not save the villager and kill/not kill the werewolf
				#
				#   Winner  |    Save     |  Not save
				# ----------+-------------+-------------
				#    Kill   | Nonwerewolf | Nonwerewolf         
				# ----------+-------------+-------------
				#  Not kill |  Not ended  |  Werewolf
				#
				game_state = self.determine_win_condition()
				# Witch need to store transition
				self.batch_update_transitions([self.witch], game_state)
			else:    # Cannot save
				self.set_player_killed(player_to_kill, False)

			player_to_witch_kill = self.witch.get_player_to_witch_kill()
			if player_to_witch_kill != -1:
				assert not self.witch.witch_killed
				if self.display_process:
					print("Witch killed {0}".format(player_to_witch_kill))
				self.witch.set_witch_killed()
				self.set_player_killed(player_to_witch_kill, False)
		else:
			self.set_player_killed(player_to_kill, False)

		game_state = self.determine_win_condition()
		self.batch_update_transitions(self.players, game_state)
		self.batch_clear_transitions(self.players)
		if game_state != GameState.NOT_ENDED:
			return game_state

		# Seer identifies a player
		if self.seer.is_alive:
			player_to_identify = self.seer.get_player_to_identify()
			if self.display_process:
				print("Seer identified {0}".format(player_to_identify))
			self.seer.identify(
				player_to_identify,
				self.players[player_to_identify].get_role()
			)

		self.utter()
		self.batch_update_transitions(self.players, GameState.NOT_ENDED)

		# Vote
		highest_voted_players = self.vote(captain=False)
		if len(highest_voted_players) == 1:    # exactly one player has the highest vote
			if self.display_process:
				print("{0} is voted to death".format(highest_voted_players[0]))
			self.set_player_killed(highest_voted_players[0], True)
		else:
			if self.display_process:
				print("Vote is undecided between {0}".format(highest_voted_players))
			highest_voted_players = self.vote(highest_voted_players)
			if len(highest_voted_players) == 1:
				if self.display_process:
					print("{0} is voted to death".format(highest_voted_players[0]))
				self.set_player_killed(highest_voted_players[0], True)
			# else: No player is voted to death this turn

		game_state = self.determine_win_condition()
		self.batch_update_transitions(self.players, game_state)
		self.batch_clear_transitions(self.werewolves)
		return game_state

def run():
	for episode in range(10):
		game.new_game()

if __name__ == "__main__":
	game = Game([
		Role.VILLAGER, Role.VILLAGER, Role.VILLAGER,
		Role.WEREWOLF, Role.WEREWOLF, Role.WEREWOLF,
		Role.SEER, Role.WITCH, Role.HUNTER
	], has_captain=False)

	game_network = GameNetwork(
		player_num=game.player_num,
		has_captain=game.has_captain
	)

	run()