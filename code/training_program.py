import argparse
import socket
import time
import os
import numpy as np
import logging
import multiprocessing as mp
from tqdm import tqdm
from keras.models import load_model, save_model
import json
import utils
from uuid import uuid4

from selfplay import setup
from game import Game
import config
from GUI.display import GUI
from train import Trainer

LOCAL_PREDICTIONS = False

PROGRAMS_FOLDER = "programs"

log_messages = []

def add_file_log(level, message):
    entry = f"{level.upper()}: {message}"
    with open(LOG_FILE, 'a') as log_file:
            log_file.write(entry + '\n')
    

class SharedCounter:
    def __init__(self, threshold):
        self.value = mp.Value('i', 0)
        self.threshold = threshold
        self.pbar = tqdm(total=threshold, desc="Games finished")

    def increment(self) -> bool:
        with self.value.get_lock():
            self.value.value += 1
            self.pbar.n = self.value.value
            self.pbar.refresh()
            logging.debug(f"Number of games finished: {self.value.value}")
            return self.value.value >= self.threshold


game_counter = None

@utils.time_function
def self_play_for(number_of_games, mem_folder, p_count = 8, experimental=False):
    
    global game_counter 
    game_counter = SharedCounter(number_of_games)
    
    if number_of_games < p_count:
        p_count=number_of_games
    
    with mp.Pool(processes=p_count) as pool:
        pool.starmap(self_play_count, [(i+1, mem_folder, experimental) for i in range(p_count)])

def self_play_count(index, mem_folder, experimental=False):
    """
    Continuously play games against itself until the limit is reached
    """
    global game_counter
    
    game = setup(local_predictions=LOCAL_PREDICTIONS,model_path=args["model"], pbar_i=index, experimental=experimental)
    game.set_mem_folder(mem_folder)

    if config.SELFPLAY_SHOW_BOARD:
        gui = GUI(400, 400, game.env.board.turn)
        game.GUI = gui
    
    working = True
    while working:
        if config.SELFPLAY_SHOW_BOARD:
            game.GUI.gameboard.board.set_fen(game.env.board.fen())
            game.GUI.draw()
        game.play_one_game(stochastic=True)
        
        if game_counter.increment():
            working = False
    
    game.white.close()
    game.black.close()
        
            
def puzzle_solver(puzzles, number_of_puzzles, memory_folder, experimental=False):
    """
    Continuously solve puzzles until the limit is reached
    """
    game = setup(local_predictions=LOCAL_PREDICTIONS, model_path=args["model"], pbar_i=1, experimental=experimental)
    game.set_mem_folder(memory_folder)
    
    # shuffle pandas rows
    puzzles = puzzles.sample(frac=1).reset_index(drop=True)
    puzzles = puzzles.head(number_of_puzzles)
    game.train_puzzles(puzzles)
    
def change_model_server(model_path: str):
    """
    Request the server to change the model to the specified model path.
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    server = config.SOCKET_HOST
    port = 5000
    s.connect((server, port))
    
    # Prepare the request payload
    request = json.dumps({
        "action": "load_model",
        "model_path": model_path
    })
    
    logging.debug(f"sending request: {request}")

    # Send the length of the request and the request itself
    s.send(f"{len(request):010d}".encode('ascii'))
    s.send(request.encode('ascii'))

    # Receive the length of the response
    response_length = s.recv(10)
    response_length = int(response_length.decode("ascii"))
    
    # Receive the response
    response = utils.recvall(s, response_length)
    s.close()
    
    response = json.loads(response.decode("ascii"))
    logging.debug(f"Received response: {response}")

    # Check the response status
    if response.get("status") == "success":
        logging.info("Succesfully switched to new model.")
    else:
        logging.error(response.get("message"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create training program for model")
    parser.add_argument("--model", type=str, help="The model to train")
    parser.add_argument("--name", type=str, help="The name of the output trained model")
    parser.add_argument("--puzzle-file", default=None, type=str, help="File to load puzzles from (csv)")
    parser.add_argument("--experimental", action="store_true")
    args = parser.parse_args()
    args = vars(args)
    
    if not LOCAL_PREDICTIONS:
        # wait until server is ready
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        server = config.SOCKET_HOST
        port = 5000

        print("Checking if server is ready...")
        while s.connect_ex((server, port)) != 0:
            print(f"Waiting for server at {server}:{port}")
            time.sleep(1)
        print(f"Server is ready on {s.getsockname()}!")
        s.close()
        
    # create model folder
    model_folder = os.path.join(PROGRAMS_FOLDER, args["name"])
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)
    
    # log file
    LOG_FILE = os.path.join(model_folder, "traininglog.txt")
    
    logging.basicConfig(level=logging.INFO,)
                    # filename=LOG_FILE,
                    # filemode='a',
                    # format='%(asctime)s %(levelname)s %(message)s')
    add_file_log("\ninfo", "Starting training")
    
    models = [args["model"]]
    add_file_log("info", f"Model '{models[-1]}' chosen")
    
    logging.debug("Checking preexisting training sets")
    i = 0
    # while os.path.exists(os.path.join(model_folder, f"tset-{i:02d}")):
    #     i+=1
    
    # creating training set
    tset_folder = os.path.join(model_folder, f"tset-{i:02d}")
    if not os.path.exists(tset_folder):
        os.makedirs(tset_folder)
    
    logging.info(f"Creating training set {i}")
    add_file_log("info", f"Creating training set {i:02d} with {config.BATCH_SIZE} games")
    # self_play_for(config.BATCH_SIZE, tset_folder, experimental=args["experimental"])
    
    if args["puzzle_file"] is not None:
        n_puzzles = 10
        add_file_log("info", f"Creating puzzle set ({n_puzzles} puzzles)")
        # puzzles = Game.create_puzzle_set(filename=args['puzzle_file'])
        # puzzle_solver(puzzles, n_puzzles, tset_folder, experimental=args["experimental"])
        
    # training model
    model = load_model(models[-1])
    trainer = Trainer(model=model)

    files = os.listdir(tset_folder + "/")
    data = []
    
    logging.info(f"Loading all games in {tset_folder}.")
    add_file_log("info", f"Loading all games in {tset_folder}")
    for file in files:
        if file.endswith('.npy'):
            data.append(np.load(f"{tset_folder}/{file}", allow_pickle=True))
    data = np.concatenate(data)

    logging.info(f"{len(data[data[:,2] > 0])} white wins, {len(data[data[:,2] < 0])} black wins, {len(data[data[:,2] == 0])} draws.")
    add_file_log("info", f"{len(data[data[:,2] > 0])} white wins, {len(data[data[:,2] < 0])} black wins, {len(data[data[:,2] == 0])} draws")
    
    # delete drawn games
    # data = data[data[:,2] != 0]
    logging.info(f"Training with {len(data)} positions")
    add_file_log("info", f"Training with {len(data)} positions")
    
    history = trainer.train_random_batches(data)
    
    # save the new model
    new_model = trainer.save_model(f"{args['name']}", it=i)
    models.append(new_model)
    logging.info(f"Saving new model '{new_model}'")
    add_file_log("info", f"Saving new model '{new_model}'")
    
    # plot
    plot_file = trainer.plot_loss(history)
    add_file_log("info", f"Exported loss plot to: {plot_file}")
    
    # this doesn't work, have to do it manually
    # change_model_server(models[-1])

        
        
        
        
            

