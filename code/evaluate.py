from agent import Agent
from chessEnv import ChessEnv
from game import Game
import config
from GUI.display import GUI
import multiprocessing as mp
from tqdm import tqdm
import socket
import time
import os

class Evaluation:
    def __init__(self, model_1_path: str, model_2_path: str, local_predictions = False):
        self.model_1 = model_1_path
        self.model_2 = model_2_path
        self.local_predictions = local_predictions
        
        eval_folder = os.path.basename(model_1_path) + "-vs-" + os.path.basename(model_2_path) 
        self.mem_folder = os.path.join("evaluations", eval_folder)
        if not os.path.exists(self.mem_folder):
            os.makedirs(self.mem_folder)

    def evaluate_games(self, progress, n, pbar_i):
        agent_1 = Agent(model_path=self.model_1, local_predictions=self.local_predictions, port=5000)
        agent_2 = Agent(model_path=self.model_2, local_predictions=self.local_predictions, port=8000)

        while progress.get("started") < n:
            progress["started"] += 1
            
            # play deterministally
            game = Game(ChessEnv(), agent_1, agent_2, pbar_i=pbar_i)
            game.set_mem_folder(self.mem_folder)

            outcome = game.play_one_game(stochastic=False)
            progress["finished"] += 1
            
            if outcome == 0:
                progress["draws"] += 1
            elif outcome == 1:
                progress["model1"] += 1
            else:
                progress["model2"] += 1
                
            # turn around the colors
            game = Game(ChessEnv(), agent_2, agent_1, pbar_i=pbar_i)
            game.set_mem_folder(self.mem_folder)

            outcome = game.play_one_game(stochastic=False)
            progress["finished"] += 1
            
            if outcome == 0:
                progress["draws"] += 1
            elif outcome == 1:
                progress["model2"] += 1
            else:
                progress["model1"] += 1
                

    def evaluate(self, n: int, p_count=8):
        """
        For n games, let the two models play each other and keep a score
        """
        if n > 0 and n < p_count:
            p_count = n
            
        if self.local_predictions:
            p_count = 1
        
        
        with mp.Manager() as manager:
            pbar_main = tqdm(total=n, desc="Total games played")
            shared_progress = manager.dict({"model1":0, "model2": 0, "draws":0, "started":0, "finished":0})
            
            with mp.Pool(processes=p_count) as pool:
                result = pool.starmap_async(self.evaluate_games, [(shared_progress, n, i+1) for i in range(p_count)])

                fin = shared_progress["finished"]
                while fin < n:
                    pbar_main.n = fin
                    pbar_main.update()
                    
                    try:
                        result.get(3)
                        break
                    except mp.TimeoutError:
                        pass
                        
                
            
            return (
                f"Evaluated these models: Model 1 = {self.model_1}, Model 2 = {self.model_2}\n"
                + f"The results: \nModel 1: {shared_progress.get('model1')} \nModel 2: {shared_progress.get('model2')} \nDraws: {shared_progress.get('draws')}"
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
    parser.add_argument("--local-predictions", action="store_true", help='Use local predictions instead of the server')
    args = parser.parse_args()
    # args to dict
    args = vars(args)
    
    if not args["local_predictions"]:
        # wait until server is ready
        s1 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s1.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        server = config.SOCKET_HOST
        port = 8000

        print("Checking if server is ready...")
        while s1.connect_ex((server, port)) != 0:
            print(f"Waiting for server at {server}:{port}")
            time.sleep(1)
        print(f"Server is ready on {s1.getsockname()}!")
        s1.close()
        
        s2 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s2.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        port = 5000

        print("Checking if server is ready...")
        while s2.connect_ex((server, port)) != 0:
            print(f"Waiting for server at {server}:{port}")
            time.sleep(1)
        print(f"Server is ready on {s2.getsockname()}!")
        s2.close()
    
    evaluation = Evaluation(args["model_1"], args["model_2"], args["local_predictions"])
    print(evaluation.evaluate(int(args["nr_games"])))
