# JSON (for reading config.json)
import json
# Asyncio (for concurrency)
import asyncio
import threading
import time

# Websockets (for communication with the server)
from websockets.client import connect, WebSocketClientProtocol
from websockets.exceptions import ConnectionClosed
from websockets.typing import Data

# Decision module
from cameraModule import CameraModule

# Import connection state object
from connectionState import ConnectionState

# Restore the ability to use Ctrl + C within asyncio
import signal
signal.signal(signal.SIGINT, signal.SIG_DFL)

# Terminal colors for formatting output text
from terminalColors import *

# GUI
import gui

# OpenCV
import cv2

# Get the connect URL from the config.json file
def getConnectURL() -> str:

	# Read the configuration file
	with open('../config.json', 'r', encoding='UTF-8') as configFile:
		config = json.load(configFile)

	# Return the websocket connect address
	return f'ws://{config["ServerIP"]}:{config["WebSocketPort"]}'

class CvClient:
	'''
	Implementation of a websocket client to communicate with the
	Pacbot game server, using asyncio.
	'''

	def __init__(self, connectURL: str, cameraIDs: list[int]) -> None:
		'''
		Construct a new Pacbot client object
		'''

		# Connection URL (starts with ws://)
		self.connectURL: str = connectURL

		# Private variable to store whether the socket is open
		self._socketOpen: bool = False

		# Connection object to communicate with the server
		self.connection: WebSocketClientProtocol

		# Event loop
		self.loop: asyncio.AbstractEventLoop

		# Game state object to store the game information
		self.state: ConnectionState = ConnectionState()

		# Decision module (policy) to make high-level decisions
		self.cameraModules = [CameraModule(self.state) for _ in cameraIDs]
		for i, module in enumerate(self.cameraModules):
			module.setCameraID(cameraIDs[i])

	async def run(self) -> None:
		'''
		Connect to the server, then run
		'''

		# Get the current event loop
		self.loop = asyncio.get_running_loop()

		# Connect to the websocket server
		await self.connect()

		try: # Try receiving messages indefinitely
			if self._socketOpen:
				await asyncio.gather(
					self.sendLoop(),
					self.receiveLoop(),
					self.displayLoop(),
					*[module.decisionLoop() for module in self.cameraModules]
				)
		finally: # Disconnect once the connection is over
			await self.disconnect()

	async def connect(self) -> None:
		'''
		Connect to the websocket server
		'''

		# Connect to the specified URL
		try:
			self.connection = await connect(self.connectURL)
			self._socketOpen = True
			self.state.setConnectionStatus(True)

		# If the connection is refused, log and return
		except ConnectionRefusedError:
			print(
				f'{RED}Websocket connection refused [{self.connectURL}]\n'
				f'Are the address and port correct, and is the '
				f'server running?{NORMAL}'
			)
			return

	async def disconnect(self) -> None:
		'''
		Disconnect from the websocket server
		'''

		# Close the connection
		if self._socketOpen:
			await self.connection.close()
		self._socketOpen = False
		self.state.setConnectionStatus(False)

		for module in self.cameraModules:
			if module.cap is not None:
				module.cap.release()

	# Return whether the connection is open
	def isOpen(self) -> bool:
		'''
		Check whether the connection is open (unused)
		'''
		return self._socketOpen

	async def sendLoop(self) -> None:
		'''
		Loop for sending messages to the server
		'''
		while self.isOpen():
			try:
				if self.state.writeServerBuf:
					response: bytes = self.state.writeServerBuf.popleft()
					await self.connection.send(response)
				await asyncio.sleep(0.01)
			except ConnectionClosed:
				break

	async def receiveLoop(self) -> None:
		'''
		Receive loop for capturing messages from the server
		'''
		try:
			async for _ in self.connection:
				pass
		except ConnectionClosed:
			print('Connection lost...')
			self.state.setConnectionStatus(False)

	def _displayWorker(self) -> None:
		'''
		Worker thread for displaying the camera feed
		'''
		# Format windows
		for i in range(len(self.cameraModules)):
			cv2.namedWindow(f"Camera {i}", cv2.WINDOW_NORMAL)
			cv2.resizeWindow(f"Camera {i}", 400, 300)

		while self.isOpen():
			for i, module in enumerate(self.cameraModules):
				if module.latest_frame is not None:
					cv2.imshow(f"Camera {i}", module.latest_frame)

			if cv2.waitKey(1) & 0xFF == ord('q'):
				asyncio.run_coroutine_threadsafe(self.disconnect(), self.loop)
				break

			time.sleep(0.01)

		cv2.destroyAllWindows()

	async def displayLoop(self) -> None:
		'''
		Loop for displaying the camera feed
		'''
		display_thread = threading.Thread(target=self._displayWorker, daemon=True)
		display_thread.start()

		while self.isOpen():
			await asyncio.sleep(0.1)

async def main():

	# Get the URL to connect to
	connectURL = getConnectURL()

	# Get camera selection
	selected_cameras = gui.get_camera_selection()
	if not selected_cameras:
		print(f"{RED}No cameras selected. Exiting.{NORMAL}")
		return

	client = CvClient(connectURL, selected_cameras)
	await client.run()

	# Once the connection is closed, end the event loop
	loop = asyncio.get_event_loop()
	loop.stop()

if __name__ == '__main__':

	# Run the event loop forever
	loop = asyncio.new_event_loop()
	loop.create_task(main())
	loop.run_forever()