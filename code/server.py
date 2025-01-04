import json
import logging
import os
import socket
import time
from tracemalloc import start
from typing import Tuple
import config
import numpy as np
import threading
import utils

# from dotenv import load_dotenv
# load_dotenv()

import tensorflow as tf
from tensorflow.keras.models import load_model

logging.basicConfig(level=logging.DEBUG, format=' %(message)s')


# Thread-safe model reference
model_lock = threading.Lock()
model = load_model(config.MODEL_FOLDER + "/brice-00.keras")

def set_model(model_path):
    """
    Load and set a new model thread-safely.
    """
    global current_model
    logging.debug("Request to change model")
    with model_lock:
        current_model = load_model(model_path)
        logging.info(f"Model loaded: {model_path}")

@tf.function(experimental_follow_type_hints=True)
def predict(args: tf.Tensor) -> Tuple[list[tf.float32], list[list[tf.float32]]]:
    """
    Make a prediction using the current model.
    """
    with model_lock:
        return model(args)

class ServerSocket:
	def __init__(self, host, port):
		"""
		The server object listens to connections and creates client handlers
		for every client (multi-threaded).

		It receives inputs from the clients and returns the predictions to the correct client.
		"""
		self.host = host
		self.port = port
  
		# first prediction
		# test_data = np.random.choice(a=[False, True], size=(1, *config.INPUT_SHAPE), p=[0, 1])
		# tf.convert_to_tensor(test_data, dtype=tf.bool)
		# p, v = predict(test_data)
		# del test_data, p, v


	def start(self):
		"""
		Start the server and listen for connections.
		"""
		logging.info(f"Starting server on {self.host}:{self.port}")
		self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
		self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
		self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
		self.sock.bind((self.host, self.port))
		# listen for incoming connections, queue up to 24 requests
		self.sock.listen(24)
		logging.info(f"Server started on {self.sock.getsockname()}")
		try:
			while True:
				self.accept()
				logging.info(f"Current thread count: {threading.active_count()}.")
		except KeyboardInterrupt:
			self.stop()
		except Exception as e:
			logging.error(f"Error: {e}")
			self.sock.close()

	def accept(self):
		"""
		Accept a connection and create a client handler for it.	
		"""
		logging.info("Waiting for client...")
		self.client, address = self.sock.accept()
		logging.info(f"Client connected from {address}")
		clh = ClientHandler(self.client, address)
		# start new thread to handle client
		clh.start()

	def stop(self):
		logging.info("Stopping server...")
		self.sock.close()
		logging.info("Server stopped.")


class ClientHandler(threading.Thread):
	def __init__(self, sock: socket.socket, address: Tuple[str, int]):
		"""
		The ClientHandler object handles a single client connection, and sends
		inputs to the server, and returns the server's predictions to the client.
		"""
		super().__init__()
		self.BUFFER_SIZE = config.SOCKET_BUFFER_SIZE
		self.sock = sock
		self.address = address

	def run(self):
		"""Create a new thread"""
		logging.info(f"ClientHandler started for {self.address}")
		while True:
			request = self.receive()
			if request is None or len(request) == 0:
				self.close()
				break

			try:
				# Parse the request
				request = json.loads(request.decode('utf-8'))

				if request.get("action") == "predict":
					data_list = request.get("data")
					# Prediction request
					# data = np.array(np.frombuffer(request.get("data"), dtype=bool))
					data = np.array(data_list, dtype=bool)
					data = data.reshape(1, *config.INPUT_SHAPE)
					tensor_data = tf.convert_to_tensor(data, dtype=tf.bool)
			
   					# make prediction
					p, v = predict(tensor_data)
					p, v = p[0].numpy().tolist(), float(v[0][0])
					response = json.dumps({"prediction": p, "value": v})

				elif request.get("action") == "load_model":
                    # Load model request
					model_path = request.get("model_path")
					logging.debug(f"Request to load model: {model_path}")
     
					if model_path and os.path.exists(model_path):
						set_model(model_path)
						response = json.dumps({"status": "success", "message": f"Model loaded: {model_path}"})
					else:
						response = json.dumps({"status": "error", "message": "Model path invalid or does not exist."})
				else:
					response = json.dumps({"status": "error", "message": "Invalid action."})

			except Exception as e:
				logging.error(f"Error handling request: {e}")
				response = json.dumps({"status": "error", "message": str(e)})
    
			self.send(f"{len(response):010d}".encode('ascii'))
			self.send(response.encode('ascii'))

	def receive(self):
		"""
		Receive data from the client.
		"""
		data = None
		try:
			data_length = self.sock.recv(10)
			if data_length == b'':
				# this happens if the socket connects and then closes without sending data
				return data
			data_length = int(data_length.decode("ascii"))
			logging.debug(f"Data length: {data_length}")
			data = utils.recvall(self.sock, data_length)
   
		except ConnectionResetError:
			logging.warning(f"Connection reset by peer. Client IP: {str(self.address[0])}:{str(self.address[1])}")
		except ValueError as e:
			logging.warning(e)
		return data

	def send(self, data):
		"""
		Send data to the client.
		"""
		logging.debug("Sending data...")
		self.sock.send(data)
		logging.debug("Data sent.")

	def close(self):
		"""
		Close the client connection.
		"""
		logging.info(f"Closing connection to {self.address}")
		self.sock.close()
		logging.debug("Connection closed.")
	

if __name__ == "__main__":
	# create the server socket and start the server
	server = ServerSocket(os.environ.get("SOCKET_HOST", "0.0.0.0"), int(os.environ.get("SOCKET_PORT", 5000)))
	server.start()
	
