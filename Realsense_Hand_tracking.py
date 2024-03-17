## License: Apache 2.0. See LICENSE file in root directory.
## Copyright(c) 2015-2017 Intel Corporation. All Rights Reserved.

###############################################
##      Open CV and Numpy integration        ##
###############################################

import pyrealsense2 as rs
import numpy as np
import mediapipe as mp
import cv2


class handdetec:
    def __init__(self, mode=False, max_hands=2, modelComplex=1, detection=0.5, tracking=0.5):
        self.mode = mode
        self.max_hands = max_hands
        self.complex = modelComplex
        self.detection = detection
        self.track = tracking

        # detecting the landmarks
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.max_hands, modelComplex, detection, tracking)
        self.mpdraw = mp.solutions.drawing_utils

    # job is to find the hands
    def find_hands(self, img, draw=True):
        self.RGBimg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.result = self.hands.process(self.RGBimg)

        if self.result.multi_hand_landmarks:
            for handlms in self.result.multi_hand_landmarks:
                if draw:
                    self.mpdraw.draw_landmarks(img, handlms, self.mpHands.HAND_CONNECTIONS)

        return img

    # find the positions (we'll use to then output)
    #need to add a z field and a way to get it
    #additionally, have the output by more horizontally
    def find_position(self, img, handNum=0, draw=True):
        lmlist = []

        if self.result.multi_hand_landmarks:
            myHand = self.result.multi_hand_landmarks[handNum]

            physical_width = 1.0 #example in meters
            physical_height = 1.0
            resolution_width, resolution_height = img.shape[1], img.shape[0]

            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)  #, int(lm.z * l)
                #convert the pixels to meters
                real_world_x, real_world_y = pixels_to_meters(cx, cy, physical_width, physical_height, resolution_width, resolution_height)

                lmlist.append([id, cx, cy, real_world_x, real_world_y])
                if draw:
                    cv2.circle(img, (cx, cy), 10, (255, 0, 0), cv2.FILLED)

        # cx = pixel x value, cy = pixel y value
        return lmlist

#converts pixels to meters
def pixels_to_meters(pixel_x, pixel_y, physical_width, physical_height, resolution_width, resolution_height):
    pixel_size_horizontal = physical_width / resolution_width
    pixel_size_veritcal = physical_height / resolution_height

    real_world_x = pixel_x * pixel_size_horizontal
    real_world_y = pixel_y * pixel_size_veritcal\

    return real_world_x, real_world_y


def main():
    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()
    detector = handdetec()

    # Get device product line for setting a supporting resolution
    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    pipeline_profile = config.resolve(pipeline_wrapper)
    device = pipeline_profile.get_device()
    device_product_line = str(device.get_info(rs.camera_info.product_line))

    found_rgb = False
    for s in device.sensors:
        if s.get_info(rs.camera_info.name) == 'RGB Camera':
            found_rgb = True
            break
    if not found_rgb:
        print("The demo requires Depth camera with Color sensor")
        exit(0)

    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    if device_product_line == 'L500':
        config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
    else:
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # Start streaming
    pipeline.start(config)

    try:
        while True:

            # Wait for a coherent pair of frames: depth and color
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue


            # Convert images to numpy arrays
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            img = detector.findhands(color_image)
            xyvals = detector.findposition(img)

            #to access depth pixel: depth_image[val[2]][val[1]] from val in xyvals
            #print(depth_image[val[2]][val[1]])

            # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

            depth_colormap_dim = depth_colormap.shape
            color_colormap_dim = color_image.shape

            # If depth and color resolutions are different, resize color image to match depth image for display
            if depth_colormap_dim != color_colormap_dim:
                resized_color_image = cv2.resize(color_image, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]),
                                                 interpolation=cv2.INTER_AREA)
                images = np.hstack((resized_color_image, depth_colormap))
            else:
                images = np.hstack((color_image, depth_colormap))

            # Show images
            cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('RealSense', images)
            cv2.waitKey(1)

    finally:

        # Stop streaming
        pipeline.stop()

"""
        # calculating time
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 1)

        cv2.imshow('img', img)
        if cv2.waitKey(20) & 0xFF == ord('d'):
            break
"""

if __name__ == '__main__':
    main()
