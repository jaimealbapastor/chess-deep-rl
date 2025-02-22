import os
import time
from chessEnv import ChessEnv
from agent import Agent
import utils
import logging
import config
from chess.pgn import Game as ChessGame
from edge import Edge
from mcts import MCTS, ExperimentalMCTS
import uuid
import pandas as pd
import numpy as np
from tqdm import tqdm

class Game:
    def __init__(self, env: ChessEnv, white: Agent, black: Agent, pbar_i=None, experimental=False):
        """
        The Game class is used to play chess games between two agents.
        """
        self.env = env
        self.white = white
        self.black = black

        self.chessgame = None
        self.memory = []
        self.memory_folder = config.MEMORY_DIR
        
        self.pbar_i = pbar_i
        self.experimental = experimental
        
        self.reset()

    def reset(self):
        self.env.reset()
        self.turn = self.env.board.turn  # True = white, False = black

    @staticmethod
    def get_winner(result: str) -> int:
        return 1 if result == "1-0" else - 1 if result == "0-1" else 0
            

    # @utils.time_function
    def play_one_game(self, stochastic: bool = True) -> int:
        """
        Play one game from the starting position, and save it to memory.
        Keep playing moves until either the game is over, or it has reached the move limit.
        If the move limit is reached, the winner is estimated.
        """
        # reset everything
        self.reset()
        # add a new memory entry
        # self.memory.append([])
        del self.memory
        self.memory = []
        # show the board
        logging.info(f"\n{self.env.board}")
        # counter to check amount of moves played. if above limit, estimate winner
        counter, previous_edges, full_game = 0, (None, None), True
        while not self.env.board.is_game_over():
            # play one move (previous move is used for updating the MCTS tree)
            previous_edges = self.play_move(stochastic=stochastic, previous_moves=previous_edges, counter=counter)
            # logging.info(f"\n{self.env.board}")
            # logging.info(f"Value according to white: {self.white.mcts.root.value}")
            # logging.info(f"Value according to black: {self.black.mcts.root.value}")
            if config.SELFPLAY_SHOW_BOARD and hasattr(self, 'GUI'):
                self.GUI.gameboard.board.set_fen(self.env.board.fen()) 
                self.GUI.draw()

            # end if the game drags on too long
            counter += 1
            if counter > config.MAX_GAME_MOVES or self.env.board.is_repetition(3):
                # estimate the winner based on piece values
                winner = ChessEnv.estimate_winner(self.env.board)
                logging.info(f"Game over by move limit ({config.MAX_GAME_MOVES}). Result: {winner}")
                full_game = False
                break
        if full_game:
            # get the winner based on the result of the game
            winner = Game.get_winner(self.env.board.result())
            logging.info(f"Game over. Result: {winner}")
        # save game result to memory for all games
        for index, element in enumerate(self.memory):
            self.memory[index] = (element[0], element[1], winner)

        self.chessgame = ChessGame()
        self.chessgame.headers["Event"] = "Evaluation"
        self.chessgame.headers["White"] = self.white.model_path
        self.chessgame.headers["Black"] = self.black.model_path
        
        # set starting position
        self.chessgame.setup(self.env.fen)
    
        # add moves
        node = self.chessgame.add_variation(self.env.board.move_stack[0])
        for move in self.env.board.move_stack[1:]:
            node = node.add_variation(move)
        # print pgn
        logging.info(self.chessgame)

        # save memory to file
        self.save_game(name="game", full_game=full_game)

        return winner
    

    def play_move(self, stochastic: bool = True, previous_moves: tuple[Edge, Edge] = (None, None), save_moves=True, counter = None) -> tuple:
        """
        Play one move. If stochastic is True, the move is chosen using a probability distribution.
        Otherwise, the move is chosen based on the highest N (deterministically).
        The previous moves are used to reuse the MCTS tree (if possible): the root node is set to the
        node found after playing the previous moves in the current tree.
        """
        # whose turn is it
        current_player = self.white if self.turn else self.black

        if previous_moves[0] is None or previous_moves[1] is None:
            # create new tree with root node == current board
            if not self.experimental:
                current_player.mcts = MCTS(current_player, state=self.env.board.fen(), stochastic=stochastic, pbar_i = self.pbar_i)
            else:
                current_player.mcts = ExperimentalMCTS(current_player, state=self.env.board.fen(), stochastic=stochastic, pbar_i=self.pbar_i)
        else:
            # change the root node to the node after playing the two previous moves
            try:
                node = current_player.mcts.root.get_edge(previous_moves[0].action).output_node
                node = node.get_edge(previous_moves[1].action).output_node 
                current_player.mcts.root = node
            except AttributeError as e:
                # logging.warning("WARN: Node does not exist in tree, continuing with new tree...")
                if not self.experimental:
                    current_player.mcts = MCTS(current_player, state=self.env.board.fen(), stochastic=stochastic, pbar_i = self.pbar_i)
                else:
                    current_player.mcts = ExperimentalMCTS(current_player, state=self.env.board.fen(), stochastic=stochastic, pbar_i=self.pbar_i)
        
        # play n simulations from the root node
        current_player.run_simulations(n=config.SIMULATIONS_PER_MOVE, step_num = counter)
        
        moves = current_player.mcts.root.edges
        if save_moves:
            self.save_to_memory(self.env.board.fen(), moves)
            
        zero_division = False
        best_move = None
                
        if self.experimental:
            t = 1 if self.turn else -1  # min max
            values = [t * e.W for e in moves]
            
            if stochastic:
                min_v = min(values)
                values = [v - min_v for v in values] # make all positive
                total = sum(values)
                if total != 0: 
                    prob_values = [v/total for v in values]
                    best_move = np.random.choice(moves, p=prob_values)
                else:
                    zero_division = True
                    
            if zero_division or not stochastic:
                t = 1 if self.turn else -1  # min max
                values = [t * e.W for e in moves]
                best_move = moves[np.argmax(values)]

        else:
            sum_move_visits = sum(e.N for e in moves)
            probs = [e.N / sum_move_visits for e in moves]
        
            if stochastic:
                # choose a move based on a probability distribution
                best_move = np.random.choice(moves, p=probs)
            else:
                # choose a move based on the highest N
                best_move = moves[np.argmax(probs)]

        # play the move
        logging.info(
            f"{'White' if self.turn else 'Black'} played  {self.env.board.fullmove_number}. {best_move.action}")
        self.env.step(best_move.action)
        
        # switch turn
        self.turn = not self.turn

        # return the previous move and the new move
        return (previous_moves[1], best_move)
    

    def save_to_memory(self, state, moves) -> None:
        """
        Append the current state and move probabilities to the internal memory.
        """
        if not self.experimental:
            sum_move_visits = sum(e.N for e in moves)
            # create dictionary of moves and their probabilities
            search_probabilities = {
                e.action.uci(): e.N / sum_move_visits for e in moves}
            
        else:
            sum_move_values = sum(e.W for e in moves)
            search_probabilities = {
                e.action.uci(): e.W / sum_move_values for e in moves}
        
        # winner gets added after game is over
            self.memory.append((state, search_probabilities, None))
        
    def set_mem_folder(self, memory_folder:str) -> None:
        self.memory_folder = memory_folder

    def save_game(self, name: str = "game", full_game: bool = False) -> None:
        """
        Save the internal memory to a .npy file.
        """
        # the game id consist of game + datetime
        game_id = f"{name}-{str(uuid.uuid4())[:8]}"
        if full_game:
            # if the game result was not estimated, save the game id to a seperate file (to look at later)
            with open(os.path.join(self.memory_folder, "full_games.txt"), "a") as f:
                f.write(f"{game_id}.npy\n")
        else:
            with open(os.path.join(self.memory_folder,"nonfinished_games.txt"), "a") as f:
                f.write(f"{game_id}.npy\n")

        np.save(os.path.join(self.memory_folder, game_id), self.memory)
        logging.info(
            f"Game saved to {os.path.join(self.memory_folder, game_id)}.npy")
        logging.info(f"Memory size: {len(self.memory)}")
        
        if self.chessgame is not None:
            pgn_folder = os.path.join(self.memory_folder, "pgn")
            if not os.path.exists(pgn_folder):
                os.makedirs(pgn_folder)
                
            with open(os.path.join(pgn_folder, game_id+".pgn"), 'w') as f:
                f.write(self.chessgame.__str__() + "\n")

    @utils.time_function
    def train_puzzles(self, puzzles: pd.DataFrame):
        """
        Create positions from puzzles (fen strings) and let the MCTS figure out how to solve them.
        The saved positions can be used to train the neural network.
        """
        logging.info(f"Training on {len(puzzles)} puzzles")
        for puzzle in tqdm(puzzles.itertuples(), desc="Puzzles finished", total=len(puzzles)):
            self.env.fen = puzzle.fen
            self.env.reset()
            
            # play the first move
            moves = puzzle.moves.split(" ")
            self.env.board.push_uci(moves.pop(0))
            logging.info(f"Puzzle to solve ({puzzle.rating} ELO): {self.env.fen}")
            logging.info(f"\n{self.env.board}")
            logging.info(f"Correct solution: {moves} ({len(moves)} moves)")
            del self.memory
            self.memory = []
            counter, previous_edges = 0, (None, None)
            while not self.env.board.is_game_over():
                # deterministically choose the next move (we want no exploration here)
                previous_edges = self.play_move(stochastic=False, previous_moves=previous_edges, counter=counter)
                logging.info(f"\n{self.env.board}")
                logging.info(f"Value according to white: {self.white.mcts.root.value}")
                logging.info(f"Value according to black: {self.black.mcts.root.value}")
                counter += 1
                if counter > config.MAX_PUZZLE_MOVES:
                    logging.warning(f"Puzzle could not be solved within the move limit")
                    break
            if not self.env.board.is_game_over():
                continue
            logging.info(f"Puzzle complete. Ended after {counter} moves: {self.env.board.result()}")
            
            # save game result to memory for all games
            winner = Game.get_winner(self.env.board.result())
            for index, element in enumerate(self.memory):
                self.memory[index] = (element[0], element[1], winner)

            game = ChessGame()
            # set starting position
            game.setup(self.env.fen)
            # add moves
            node = game.add_variation(self.env.board.move_stack[0])
            for move in self.env.board.move_stack[1:]:
                logging.info(move)
                node = node.add_variation(move)
            # print pgn
            logging.info(game)

            # save memory to file
            self.save_game(name="puzzle")

    @staticmethod
    def create_puzzle_set(filename: str, type: str = "mateIn2") -> pd.DataFrame:
        """
        Load the puzzles from a csv file. The type of puzzle can be specified.
        Return the puzzles as a Pandas DataFrame.
        """
        start_time = time.time()
        types={"PuzzleId": str, "FEN": str, "Moves": str, "Rating":int, "RatingDeviation":int, "Popularity":int, "NbPlays":int, "Themes":str, "GameUrl":str, "OpeningTags":str}
        puzzles: pd.DataFrame = pd.read_csv(filename, header=None, dtype=types)
        # drop unnecessary columns
        puzzles = puzzles.drop(columns=[0, 4, 5, 6, 8, 9])
        # set column names
        puzzles.columns = ["fen", "moves", "rating", "type"]
        # only keep puzzles where type contains "mate"
        puzzles = puzzles[puzzles["type"].str.contains(type)]
        logging.info(f"Created puzzles in {time.time() - start_time} seconds")
        return puzzles

    