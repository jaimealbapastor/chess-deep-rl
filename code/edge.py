from chess import Move
import config
import math
import chess
from collections import Counter
import random
import numpy as np

class Edge:
    def __init__(self, input_node: "Node", output_node: "Node", action: Move, prior: float):
        self.input_node = input_node
        self.output_node = output_node
        self.action = action

        self.player_turn = self.input_node.state.split(" ")[1] == "w"

        # each action stores 4 numbers:
        self.N = 0  # amount of times this action has been taken (=visit count)
        self.W = 0  # total action-value
        self.P = prior  # prior probability of selecting this action

    def __eq__(self, edge: object) -> bool:
        if isinstance(edge, Edge):
            return self.action == edge.action and self.input_node.state == edge.input_node.state
        else:
            return NotImplemented

    def __str__(self):
        return f"{self.action.uci()}: Q={self.W / self.N if self.N != 0 else 0}, N={self.N}, W={self.W}, P={self.P}, U = {self.upper_confidence_bound()}"

    def __repr__(self):
        return f"{self.action.uci()}: Q={self.W / self.N if self.N != 0 else 0}, N={self.N}, W={self.W}, P={self.P}, U = {self.upper_confidence_bound()}"

    def upper_confidence_bound(self, noise: float) -> float:
        exploration_rate = math.log((1 + self.input_node.N + config.C_base) / config.C_base) + config.C_init
        ucb = exploration_rate * (self.P * noise) * (math.sqrt(self.input_node.N) / (1 + self.N))
        if self.input_node.turn == chess.WHITE:
            return self.W / (self.N + 1) + ucb 
        else:
            return -(self.W / (self.N + 1)) + ucb
        
    def fast_ucb(self, noise:float) -> float:
        if self.input_node.turn == chess.WHITE:
            return self.W / (self.N + 1) + noise
        else:
            return -(self.W / (self.N + 1)) + noise
    
    def epsilon_greedy(self, epsilon=0.3):
        # Greedy epsilon
        best_edge = None
        best_score = -np.inf 
        if random.random() < epsilon:
            best_edge = random.choice(self.input_node.edges)
        else:
            for edge in self.input_node.edges:
                turn = 1 if self.input_node.turn == chess.WHITE else -1
                if (turn * edge.W) > best_score:    # min max
                    best_score = edge.W
                    best_edge = edge  
        return best_edge
        
    def reward(self)->float:
        """Returns the differnce value of pieces. Positive for white, negative for black."""
        
        outcome = chess.Board(self.output_node.state).outcome()
        if outcome:
            return 1 if outcome.winner else -1
        
        input_fen:str = self.input_node.state.split()[0]
        input_fen = ''.join([p for p in input_fen if p.isalpha()])
        counter_input = Counter(input_fen)
        
        output_fen:str = self.input_node.state.split()[0]
        output_fen = ''.join([p for p in output_fen if p.isalpha()])
        counter_output = Counter(output_fen)
        
        dif = dict(counter_input - counter_output)
        value_per_piece = {
            'p':0.1, 'P':-0.1,
            'n':0.3, 'N':-0.3,
            'b':0.3, 'B':-0.3,
            'r':0.5, 'R':-0.5,
            'q':0.9,'Q':-0.9
        }
        
        # -0.1 to encourage fast wins
        reward_value = -0.001 if self.output_node.turn else 0.001
        
        # different pieces
        for key, value in dif.items():
            reward_value += value_per_piece.get(key)
        
        return reward_value