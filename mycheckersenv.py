import functools
import numpy as np

from gymnasium.spaces import Discrete, Box
from pettingzoo import AECEnv
from pettingzoo.utils import AgentSelector


BOARD_SIZE = 6
NUM_ACTIONS = BOARD_SIZE * BOARD_SIZE * 4

EMPTY = 0
P0 = 1
P0_K = 2
P1 = -1
P1_K = -2

DIRECTIONS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]


def env():
    return raw_env()


class raw_env(AECEnv):
    metadata = {"name": "checkers_6x6_v0", "render_modes": ["human"]}

    def __init__(self):
        self.possible_agents = ["player_0", "player_1"]
        self.agent_name_mapping = dict(zip(self.possible_agents, [0, 1]))

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return Box(low=-2, high=2, shape=(BOARD_SIZE * BOARD_SIZE + 1,), dtype=np.int8)

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return Discrete(NUM_ACTIONS)

    def reset(self, seed=None, options=None):
        self.agents = self.possible_agents[:]

        self.board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=int)

        # initialize board
        for r in range(2):
            for c in range(BOARD_SIZE):
                if (r + c) % 2 == 1:
                    self.board[r][c] = P1

        for r in range(BOARD_SIZE - 2, BOARD_SIZE):
            for c in range(BOARD_SIZE):
                if (r + c) % 2 == 1:
                    self.board[r][c] = P0

        self.rewards = {a: 0 for a in self.agents}
        self._cumulative_rewards = {a: 0 for a in self.agents}
        self.terminations = {a: False for a in self.agents}
        self.truncations = {a: False for a in self.agents}
        self.infos = {a: {} for a in self.agents}

        self._agent_selector = AgentSelector(self.agents)
        self.agent_selection = self._agent_selector.next()

    def observe(self, agent):
        player_id = self.agent_name_mapping[agent]
        return np.append(self.board.flatten(), player_id).astype(np.int8)

    def _get_valid_moves(self, player):
        moves = []
        captures = []

        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                piece = self.board[r][c]

                if player == 0 and piece <= 0:
                    continue
                if player == 1 and piece >= 0:
                    continue

                is_king = abs(piece) == 2

                for d_idx, (dr, dc) in enumerate(DIRECTIONS):

                    nr, nc = r + dr, c + dc
                    nr2, nc2 = r + 2 * dr, c + 2 * dc

                    # 🔥 CAPTURE (allowed in ALL directions)
                    if (
                        0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE
                        and 0 <= nr2 < BOARD_SIZE and 0 <= nc2 < BOARD_SIZE
                    ):
                        if self.board[nr][nc] * piece < 0 and self.board[nr2][nc2] == EMPTY:
                            captures.append((r, c, d_idx))

                    # 🔹 NORMAL MOVE (direction restricted)
                    allow_normal = True
                    if not is_king:
                        if player == 0 and dr > 0:
                            allow_normal = False
                        if player == 1 and dr < 0:
                            allow_normal = False

                    if allow_normal:
                        if (
                            0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE
                            and self.board[nr][nc] == EMPTY
                        ):
                            moves.append((r, c, d_idx))

        # mandatory capture
        return captures if len(captures) > 0 else moves

    def step(self, action):
        agent = self.agent_selection

        if self.terminations[agent]:
            self._was_dead_step(action)
            return

        player = self.agent_name_mapping[agent]
        self._cumulative_rewards[agent] = 0

        from_cell = action // 4
        direction = action % 4

        r = from_cell // BOARD_SIZE
        c = from_cell % BOARD_SIZE

        valid_moves = self._get_valid_moves(player)

        if (r, c, direction) not in valid_moves:
            self.rewards[agent] = -1
        else:
            dr, dc = DIRECTIONS[direction]
            piece = self.board[r][c]

            nr_mid, nc_mid = r + dr, c + dc
            nr2, nc2 = r + 2 * dr, c + 2 * dc

            # check capture
            if (
                0 <= nr_mid < BOARD_SIZE and 0 <= nc_mid < BOARD_SIZE
                and 0 <= nr2 < BOARD_SIZE and 0 <= nc2 < BOARD_SIZE
                and self.board[nr_mid][nc_mid] * piece < 0
                and self.board[nr2][nc2] == EMPTY
            ):
                nr, nc = nr2, nc2
                self.board[nr_mid][nc_mid] = EMPTY
                self.rewards[agent] = 0.1
            else:
                nr, nc = r + dr, c + dc
                self.rewards[agent] = 0

            self.board[r][c] = EMPTY
            self.board[nr][nc] = piece

            # king promotion
            if player == 0 and nr == 0:
                self.board[nr][nc] = P0_K
            if player == 1 and nr == BOARD_SIZE - 1:
                self.board[nr][nc] = P1_K

        # check win
        opponent = 1 - player

        opponent_pieces = (
            np.sum(self.board < 0) if player == 0 else np.sum(self.board > 0)
        )
        opponent_moves = self._get_valid_moves(opponent)

        if opponent_pieces == 0 or len(opponent_moves) == 0:
            self.rewards[agent] = 1
            other = self.possible_agents[opponent]
            self.rewards[other] = -1
            self.terminations = {a: True for a in self.agents}

        self.agent_selection = self._agent_selector.next()
        self._accumulate_rewards()

    def render(self):
        print(self.board)

    def close(self):
        pass