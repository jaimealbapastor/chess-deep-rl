from agent import Agent
from chessEnv import ChessEnv
from game import Game
import config
from GUI.display import GUI
import multiprocessing as mp
from tqdm import tqdm

class EvaluationProgress:
    def __init__(self):
        self.model1_wins = mp.Value("i", 0)
        self.model2_wins = mp.Value("i", 0)
        self.draws = mp.Value("i", 0)
        
        self.total = mp.Value("i", 0)
        self.pbar = tqdm(desc="Games finished")
        
        self.started = mp.Value("i", 0)

    def increment(self, winner:str):
        """The parameter winner can be 'model1', 'model2' or 'draw'"""
        match winner:
            case "model1":
                with self.model1_wins.get_lock():
                    self.model1_wins.value += 1
            case "model2":
                with self.model2_wins.get_lock():
                    self.model2_wins.value += 1
            case "draw":
                with self.draws.get_lock():
                    self.draws.value += 1
            case _:
                return
        
        with self.total.get_lock():
            self.total.value += 1
            self.pbar.n = self.total.value
            self.pbar.refresh()
    
    def get(self, player:str):
        """The parameter player can be 'model1', 'model2', 'draw', 'total' or 'started'"""
        match player:
            case "model1":
                with self.model1_wins.get_lock():
                    return self.model1_wins.value
            case "model2":
                with self.model2_wins.get_lock():
                    return self.model2_wins.value
            case "draw":
                with self.draws.get_lock():
                    return self.draws.value
            case "total":
                with self.total.get_lock():
                    return self.total.value
            case "started":
                with self.started.get_lock():
                    return self.started.value
                
    def increment_by_outcome(self, outcome:int):
        if outcome == 0:
            self.increment("draw")
        elif outcome == 1:
            self.increment("model1")
        else:
            self.increment("model2")
            
    def start(self):
        with self.started.get_lock():
            self.started.value += 1
        
        
        
        

class Evaluation:
    def __init__(self, model_1_path: str, model_2_path: str):
        self.model_1 = model_1_path
        self.model_2 = model_2_path
        
        self.progress = EvaluationProgress()
        self.number_of_games = 1

    def evaluate_games(self, p_bar=None, n=1):
        agent_1 = Agent(local_predictions=True, model_path=self.model_1, pbar_i=p_bar)
        agent_2 = Agent(local_predictions=True, model_path=self.model_2, pbar_i=p_bar)

        while self.progress.get("started") < n:
            # play deterministally
            game = Game(ChessEnv(), agent_1, agent_2)
            game.set_mem_folder("evaluations")
            
            result = game.play_one_game(stochastic=False)
            self.progress.increment_by_outcome(result)

            # turn around the colors
            game = Game(ChessEnv(), agent_2, agent_1)
            game.set_mem_folder("evaluations")
            
            result = game.play_one_game(stochastic=False)
            self.progress.increment_by_outcome(result)

    def evaluate(self, n: int, p_count=8):
        """
        For n games, let the two models play each other and keep a score
        """
        self.number_of_games = n
        if n > 0 and n < p_count: 
            p_count=n
        
        with mp.Pool(processes=p_count) as pool:
            pool.starmap(self.evaluate_games, [i+1 for i in range(p_count)])

        return (
            f"Evaluated these models: Model 1 = {self.model_1}, Model 2 = {self.model_2}\n"
            + f"The results: \nModel 1: {self.progress.get('model1')} \nModel 2: {self.progress.get('model2')} \nDraws: {self.progress.get('draw')}"
        )


if __name__ == "__main__":
    # get args
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate two models")
    parser.add_argument("model_1", help="Path to model 1", type=str)
    parser.add_argument("model_2", help="Path to model 2", type=str)
    parser.add_argument(
        "nr_games",
        help="Number of games to play (x2: every model plays both white and black)",
        type=int,
    )
    args = parser.parse_args()

    # args to dict
    args = vars(args)

    evaluation = Evaluation(args["model_1"], args["model_2"])
    print(evaluation.evaluate(int(args["nr_games"])))
