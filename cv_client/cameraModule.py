# Asyncio (for concurrency)
import asyncio

# Import connection state object
from connectionState import ConnectionState

# Import the wall array
from walls import wallArr

# OpenCV
import cv2

# ArUco
from cv2 import aruco

# Numpy
import numpy as np

# Plt
# import matplotlib.pyplot as plt

# Typing
from typing import Any

# Extra imports for bufferless VideoCapture
import threading
import queue

# Typedef
MatLike = cv2.typing.MatLike
IntArray = np.ndarray[Any, np.dtype[np.intp]]

# Bufferless VideoCapture
class VideoCapture:

    ''' Copied from StackOverflow: https://stackoverflow.com/a/54755738 '''

    def __init__(self, name: Any):
        # Open camera (Auto backend for compatibility)
        self.cap = cv2.VideoCapture(name)
        self._stop = threading.Event()
        
        # Set properties for MJPG at 640x480 (High FPS, tolerating warnings)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1080)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not self.cap.isOpened():
            print(f"ERR: Camera {name} could not be opened.")

        self.q: queue.Queue[MatLike] = queue.Queue()
        self.t = threading.Thread(target=self._reader)
        self.t.daemon = True
        self.t.start()

    def _reader(self):
        while not self._stop.is_set():
            ret, frame = self.cap.read()
            if not ret:
                self._stop.set()
                break
            if not self.q.empty():
                try:
                    self.q.get_nowait()   # discard previous (unprocessed) frame
                except queue.Empty:
                    pass
            self.q.put(frame)

    def read(self):
        try:
            return self.q.get_nowait()
        except queue.Empty:
            return None

    def release(self):
        self._stop.set()
        self.t.join()
        self.cap.release()

class CameraModule:
    '''
    Sample implementation of a decision module for computer vision
    for Pacbot, using asyncio.
    '''

    def __init__(self, state: ConnectionState) -> None:
        '''
        Construct a new decision module object
        '''

        # Game state object to store the game information
        self.state = state

        # A dictionary of 4x4 ArUco markers
        self.dictionary = aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)

        # Instantiate a new ArUco detector
        self.detector = aruco.ArucoDetector(self.dictionary, aruco.DetectorParameters())

        # Capture object
        self.cap: VideoCapture | None = None

    def setCameraID(self, cameraID: int) -> None:
        '''
        Set the camera ID, and open the camera
        '''
        self.cap = VideoCapture(cameraID)

        # Latest frame (for display)
        self.latest_frame: MatLike | None = None


    async def decisionLoop(self) -> None:
        '''
        Decision loop for CV
        '''

        # Receive values as long as we have access
        while self.state.isConnected():

            # Get a frame
            img = self.capture()

            # If the image is none, continue
            if img is None:
                await asyncio.sleep(0.01) # prevent busy loop
                continue

            # Process the frame
            pacman_row, pacman_col = self.localize(img, annotate=True)

            # If there's a wall where the Pacbot is, quit
            if self.wallAt(pacman_row, pacman_col):
                await asyncio.sleep(0)
                continue

            # Write back to the server, as a test (move right)
            self.state.send(pacman_row, pacman_col)

            # Free up the event loop
            await asyncio.sleep(0)

    def capture(self) -> MatLike | None:
        '''
        Capture an image
        '''

        if self.cap is None:
            return None

        img = self.cap.read()
        if img is None:                                                              # type: ignore
            print("ERR: NO IMAGE")
            return None
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def wallAt(self, row: int, col: int) -> bool:
        '''
        Helper function to check if a wall is at a given location
        '''

        # Check if the position is off the grid, and return true if so
        if (row < 0 or row >= 31) or (col < 0 or col >= 28):
            return True

        # Return whether there is a wall at the location
        return bool((wallArr[row] >> col) & 1)

    def localize(self, img: MatLike, warp: bool = False, annotate: bool = False) -> tuple[int, int]:

        # Detect markers
        corners, ids, _ = self.detector.detectMarkers(img)

        if ids is None:                                                              # type: ignore
            self.latest_frame = img
            return 32, 32

        # print(ids)

        # Array of ids with centroids
        ids_centroids: list[tuple[int, IntArray]] = []

        pacman_id_found = False
        pacman_centroid = None

        # Loop over the ids
        for j in range(len(ids)):

            # Find this id
            id = ids[j, 0]

            # If the id is invalid, skip it
            if id > 6:
                continue

            # If the id is 0, Pacman has been found
            if id == 0:
                pacman_id_found = True

            # Find the coordinates of this centroid
            centroid = np.array([
                int(corners[j][0][:, 0].mean()),
                int(corners[j][0][:, 1].mean())
            ])

            # Put these together as a pair
            pair = (id, centroid)

            # Find the coordinates of each centroid
            ids_centroids.append(pair)

        # Separate Pacman's centroid if found
        if pacman_id_found:
            pacman_centroids_list = [c for i, c in ids_centroids if i == 0]
            if pacman_centroids_list:
                pacman_centroid = pacman_centroids_list[0]
            ids_centroids = [pair for pair in ids_centroids if pair[0] != 0]

        # Sort the remaining centroids (non-Pacman markers) by their IDs
        ids_centroids.sort(key=lambda x: x[0])

        # Get the sorted ids and centroids (excluding Pacman)
        corner_ids, corner_centroids = list(zip(*ids_centroids)) if ids_centroids else ((), ())

        # Initialize pacman_row and pacman_col to default "not found" values
        pacman_row, pacman_col = 32, 32

        # Define expected marker sets for each half
        top_half_expected_corner_ids = {1, 2, 3, 4}
        bottom_half_expected_corner_ids = {3, 4, 5, 6}

        top_matches = 0
        bottom_matches = 0
        is_topHalf = False
        is_bottomHalf = False

        # Create a dictionary for quick lookup of centroids by ID for corner markers
        corner_marker_map = {id: centroid for id, centroid in ids_centroids if id != 0}

        # Count matches for top and bottom halves
        for corner_id in corner_marker_map.keys():
            if corner_id in top_half_expected_corner_ids:
                top_matches += 1
            if corner_id in bottom_half_expected_corner_ids:
                bottom_matches += 1

        # Determine the dominant half based on matches, prioritizing top half in case of a tie
        if top_matches >= 4 and top_matches >= bottom_matches:
            is_topHalf = True
        elif bottom_matches >= 4 and bottom_matches > top_matches:
            is_bottomHalf = True
        
        # Initialize matrix and inverse to None
        matrix = None
        inverse = None
        offset = 0
        width = 28
        height = 0

        if is_topHalf or is_bottomHalf:
            # Collect the centroids for the four corners of the identified half
            four_corners_src = []
            expected_corner_ids_for_half = []
            if is_topHalf:
                expected_corner_ids_for_half = [1, 2, 3, 4]
                height = 16
                offset = 0
            elif is_bottomHalf:
                expected_corner_ids_for_half = [3, 4, 5, 6]
                height = 15
                offset = 16

            # Filter for only the *detected* expected corners for this half
            detected_corners_for_half_map = {id: corner_marker_map[id] for id in expected_corner_ids_for_half if id in corner_marker_map}
            
            # Sort detected corners by their IDs to maintain a consistent order for perspective transform
            sorted_detected_corners = sorted(detected_corners_for_half_map.items())

            # We need exactly 4 points to calculate a perspective transform
            if len(sorted_detected_corners) == 4:
                # Extract centroids in the sorted order
                four_corners_src = np.array([item[1] for item in sorted_detected_corners]).astype('float32')

                # Create an array describing the final locations of those points
                result = 100 * np.array([
                    [0, 0],  # Top-left
                    [width, 0],  # Top-right
                    [0, height],  # Bottom-left
                    [width, height]  # Bottom-right
                ]).astype('float32')

                # Calculate the perspective matrix
                matrix = cv2.getPerspectiveTransform(four_corners_src, result)
                inverse = np.linalg.inv(matrix) # type: ignore

                # Warp due to the perspective change
                if warp:
                    warped = cv2.warpPerspective(img, matrix, (width * 100, height * 100))
                    self.latest_frame = warped


        display_img = img.copy()
        
        if annotate:
            # Draw all detected marker centroids
            for _id, centroid in ids_centroids:
                cv2.circle(display_img, (centroid[0], centroid[1]), 5, (0, 255, 0), -1) # Green for corner markers
            if pacman_centroid is not None:
                cv2.circle(display_img, (pacman_centroid[0], pacman_centroid[1]), 5, (0, 255, 255), -1) # Yellow for Pacman

        if matrix is not None and inverse is not None:
            if annotate:
                # plt.imshow(img, cmap='gray')                                             # type: ignore
                for transformed_row in range(0, height):
                    start_point = None
                    end_point = None
                    for transformed_col in range(0, width):
                        vector = inverse @ np.array([                                    # type: ignore
                            transformed_col * 100 + 50, transformed_row * 100 + 50, 1
                        ])
                        
                        point = (int(vector[0]/vector[2]), int(vector[1]/vector[2]))
                        
                        # color = (255, 0, 255) if self.wallAt(transformed_row + offset, transformed_col) else (0, 255, 255)
                        # cv2.circle(display_img, point, 2, color, -1)

                        if self.wallAt(transformed_row + offset, transformed_col):
                            if start_point is None:
                                start_point = point
                            end_point = point
                        else:
                            if start_point is not None and end_point is not None:
                                cv2.line(display_img, start_point, end_point, (255, 0, 255), 2)
                            start_point = None
                            end_point = None
                            cv2.circle(display_img, point, 2, (0, 255, 255), -1)

            # Figure out where Pacman is (if found and transform is available)
            if pacman_centroid is not None:
                vector = matrix @ np.array([pacman_centroid[0], pacman_centroid[1], 1])

                # Figure out the transformed centroid of Pacman
                pacman_transformed_rowf = vector[1]/vector[2]/100.0 - 0.5
                pacman_transformed_colf = vector[0]/vector[2]/100.0 - 0.5

                # Round to the nearest transformed row and column
                pacman_transformed_rowr = round(pacman_transformed_rowf)
                pacman_transformed_colr = round(pacman_transformed_colf)
                # print(pacman_transformed_rowr + offset, pacman_transformed_colr, end=' -> ')

                # Loop over a 3x3 square focused on the spot
                neighbors: list[tuple[float, tuple[int, int]]] = []
                for transformed_row in range(pacman_transformed_rowr - 1, pacman_transformed_rowr + 2):
                    for transformed_col in range(pacman_transformed_colr - 1, pacman_transformed_colr + 2):
                        if not self.wallAt(transformed_row + offset, transformed_col):
                            distSq = (transformed_row - pacman_transformed_rowf) * \
                                        (transformed_row - pacman_transformed_rowf) + \
                                    (transformed_col - pacman_transformed_colf) * \
                                        (transformed_col - pacman_transformed_colf)
                            neighbors.append((distSq, (transformed_row + offset, transformed_col)))

                if not len(neighbors):
                    # print("ERR: Pacbot was found to be in a wall")
                    self.latest_frame = display_img
                    return 32, 32

                pacman_transformed_row, pacman_transformed_col = min(neighbors)[1]
                # print(pacman_transformed_row, pacman_transformed_col)
                if annotate:
                    vector = inverse @ np.array([                                            # type: ignore
                        pacman_transformed_col * 100 + 50, (pacman_transformed_row - offset) * 100 + 50, 1
                    ])
                    # plt.plot([vector[0]/vector[2]], [vector[1]/vector[2]], 'y*')                     # type: ignore
                    point = (int(vector[0]/vector[2]), int(vector[1]/vector[2]))
                    cv2.drawMarker(display_img, point, (0, 255, 255), markerType=cv2.MARKER_STAR, markerSize=10, thickness=2)

        self.latest_frame = display_img
        return pacman_row, pacman_col


