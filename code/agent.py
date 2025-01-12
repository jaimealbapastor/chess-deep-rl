import logging
import socket
from rlmodelbuilder import RLModelBuilder
import config
from keras.models import Model
import time
# import tensorflow as tf
import utils
from tqdm import tqdm
from mcts import MCTS
# from tensorflow.keras.models import load_model
import json
import numpy as np
import chess

import os
# from dotenv import load_dotenv
# load_dotenv()

class Agent:
    def __init__(self, local_predictions: bool = False, model_path = None, state=chess.STARTING_FEN, pbar_i = None):
        """
        An agent is an object that can play chessmoves on the environment.
        Based on the parameters, it can play with a local model, or send its input to a server.
        It holds an MCTS object that is used to run MCTS simulations to build a tree.
        """
        self.model_path = model_path
        if local_predictions and model_path is not None:
            logging.info("Using local predictions")
            from tensorflow.python.ops.numpy_ops import np_config
            import tensorflow as tf
            from tensorflow.keras.models import load_model
            self.model = load_model(model_path)
            self.local_predictions = True
            np_config.enable_numpy_behavior()
        else:
            logging.info("Using server predictions")
            self.local_predictions = False
            # connect to the server to do predictions
            try: 
                self.socket_to_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.socket_to_server.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                server = os.environ.get("SOCKET_HOST", "localhost")
                port = int(os.environ.get("SOCKET_PORT", 5000))
                self.socket_to_server.connect((server, port))
            except Exception as e:
                print(f"Agent could not connect to the server at {server}:{port}: ", e)
                exit(1)
            logging.info(f"Agent connected to server {server}:{port}")

        self.mcts = MCTS(self, state=state, pbar_i=pbar_i)
        

    def build_model(self) -> Model:
        """
        Build a new model based on the configuration in config.py
        """
        model_builder = RLModelBuilder(config.INPUT_SHAPE, config.OUTPUT_SHAPE)
        model = model_builder.build_model()
        return model

    def run_simulations(self, n: int = 1, step_num = None):
        """
        Run n simulations of the MCTS algorithm. This function gets called every move.
        """
        # print(f"Running {n} simulations...")
        self.mcts.run_simulations(n, step_num)

    def save_model(self, timestamped: bool = False):
        """
        Save the current model to a file
        """
        if timestamped:
            self.model.save(f"{config.MODEL_FOLDER}/model-{time.time()}.{config.MODEL_FORMAT}")
        else:
            self.model.save(f"{config.MODEL_FOLDER}/model.{config.MODEL_FORMAT}")

    def predict(self, data):
        """
        Predict locally or using the server, depending on the configuration
        """
        if self.local_predictions:
            # use tf.function
            import local_prediction
            p, v = local_prediction.predict_local(self.model, data)
            return p.numpy(), v[0][0]
        return self.predict_server(data)

    def predict_server(self, data: np.ndarray):
        """
        Send data to the server and get the prediction
        """
        
        logging.debug(f"Data to send: {data}")
        
        # Prepare the request payload
        request = json.dumps({
            "action": "predict",
            "data": data.flatten().tolist()
        }).encode('ascii')
        # Send data to server
        # self.socket_to_server.send(f"{len(request):010d}".encode('ascii'))
        self.socket_to_server.send(f"{len(request):010d}".encode('ascii'))
        self.socket_to_server.send(request)
        
        try:
            # Receive the response length
            response_length = self.socket_to_server.recv(10)
            if not response_length:
                raise ConnectionError("Server closed the connection or sent an empty response length.")
            
            response_length = int(response_length.decode("ascii"))
            
            # Receive the full response
            response = utils.recvall(self.socket_to_server, response_length)
            if not response:
                raise ConnectionError("Server closed the connection or sent an empty response.")
            
            # Decode and parse the response
            response = json.loads(response.decode("ascii"))
            
            # Check for errors in the response
            if response.get("status") == "error":
                raise ValueError(f"Server responded with an error: {response.get('message')}")
            
            # Handle the successful response for a prediction request
            if "prediction" in response and "value" in response:
                return np.array(response["prediction"]), response["value"]
            else:
                raise ValueError("Unexpected response format from the server.")

        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to decode JSON response: {e}")
        except ConnectionError as e:
            logging.error(f"Connection error: {e}")
            raise
        except ValueError as e:
            logging.error(f"Value error: {e}")
            raise
        except Exception as e:
            logging.error(f"Unexpected error: {e}")
            raise
        
    def close(self):
        if self.socket_to_server:
            self.socket_to_server.close()
            




