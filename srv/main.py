import argparse
import asyncio
import json
import logging
import os
import ssl
import math

import aiohttp_cors
import cv2
from aiohttp import web
from aiortc import MediaStreamTrack, RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaRelay
from av import VideoFrame

import dlib
import numpy as np

from scipy.interpolate import RBFInterpolator
from numpy.linalg import norm
from scipy.spatial import distance as dist
from collections import OrderedDict

# https://github.com/aiortc/aiortc/blob/main/examples/server/server.py
APP_ID = "GAZE"
ROOT = os.path.dirname(__file__)
logger = logging.getLogger(APP_ID)
pcs = set()
use_cuda = False
relay = MediaRelay()
face_detector = dlib.get_frontal_face_detector()
if use_cuda:
    face_detector = dlib.cnn_face_detection_model_v1(
        ROOT + '/model_face_detector.dat')
face_predictor = dlib.shape_predictor(ROOT + '/model_face_68_gtx_lmks.dat')
leye_pose_model = dlib.shape_predictor(ROOT + '/model_left_eye_16_lmks.dat')
reye_pose_model = dlib.shape_predictor(ROOT + '/model_rght_eye_16_lmks.dat')
# For dlib’s 68-point facial landmark detector:
face_salient_num = 68
# For dlib’s 16-point eye landmark detector:
eye_salient_num = 16
# For dlib’s 68-point facial landmark detector:
facial_landmarks = OrderedDict([
    ("mouth", (48, 68)),
    ("inner_mouth", (60, 68)),
    ("right_eyebrow", (17, 22)),
    ("left_eyebrow", (22, 27)),
    ("right_eye", (36, 42)),
    ("left_eye", (42, 48)),
    ("nose", (27, 36)),
    ("jaw", (0, 17))
])
# 3D model points.
facial_model_points = np.array([
    # Nose tip
    (0.0, 0.0, 0.0),
    # Chin
    (0.0, -330.0, -65.0),
    # Left eye left corner
    (-225.0, 170.0, -135.0),
    # Right eye right corne
    (225.0, 170.0, -135.0),
    # Left Mouth corner
    (-150.0, -150.0, -125.0),
    # Right mouth corner
    (150.0, -150.0, -125.0)
])
# For processing - frame size
resize_in_to_width = 1280
resize_in_to_height = 720
# For response - frame size
resize_out_to_width = 960
resize_out_to_height = 540


class AlphaBetaFilter:

    def __init__(self, alpha, beta, initial_state, initial_velocity, facial_model_points):
        self.alpha = alpha
        self.beta = beta
        self.state = initial_state
        self.velocity = initial_velocity
        self.facial_model_points = facial_model_points

    def update(self, measurement):
        # prediction step
        predicted_state = (self.state[0] + self.velocity[0],
                           self.state[1] + self.velocity[1])
        predicted_velocity = self.velocity

        # update step
        residual = (measurement[0] - predicted_state[0],
                    measurement[1] - predicted_state[1])
        self.state = (predicted_state[0] + self.alpha * residual[0],
                      predicted_state[1] + self.alpha * residual[1])
        self.velocity = (predicted_velocity[0] + self.beta * residual[0],
                         predicted_velocity[1] + self.beta * residual[1])

        return self.state


class EyeTrackingEngine(MediaStreamTrack):

    kind = "video"  # don't forget this!

    def __init__(self, track, face_detector, face_predictor, facial_landmarks, facial_model_points):

        super().__init__()  # don't forget this!
        self.track = track  # don't forget this!

        self.face_detector = face_detector
        self.face_predictor = face_predictor
        self.leye_pose_model = leye_pose_model
        self.reye_pose_model = reye_pose_model

        self.facial_landmarks = facial_landmarks
        self.facial_model_points = facial_model_points

        self.face_rects = None
        self.face_fails = 0
        self.face_fails_max = 3
        self.frame_index = 0
        self.frame_brightness = 0

        self.leye_eyebrow_treshold = 0.32
        self.reye_eyebrow_treshold = 0.32
        self.leye_ear_treshold = 0.22
        self.reye_ear_treshold = 0.22

        self.leye_gaze_model = None
        self.leye_calculated_point = [-1, -1]

        self.reye_gaze_model = None
        self.reye_calculated_point = [-1, -1]

        self.draw = True

        # AlphaBetaFilter parameters:
        # alpha - This parameter determines how much weight is given to the measured state. The closer it is to 1, the more weight is given to the actual measured value, meaning the filter is more sensitive to changes in the measurement.
        # beta - This parameter determines how much weight is given to the measured velocity or change in state. The closer it is to 1, the more weight is given to changes in the measurement, meaning the filter is more sensitive to changes in the velocity.

        self.abf_alpha_face = 0.5
        self.abf_beta_face = 0.3

        self.abf_alpha_eye = 0.5
        self.abf_beta_eye = 0.3

        self.abf_alpha_pose = 0.5
        self.abf_beta_pose = 0.3

        self.abf_initial_state = (0, 0)
        self.abf_initial_velocity = (0, 0)

        # initialize an alpha-beta filter for each of the face_salient_num face points
        self.abf_filter_face = [AlphaBetaFilter(self.abf_alpha_face, self.abf_beta_face, initial_state=self.abf_initial_state,
                                                initial_velocity=self.abf_initial_velocity, facial_model_points=self.facial_model_points)
                                for _ in range(face_salient_num)]

        # initialize an alpha-beta filter for each of the eye_salient_num leye points
        self.abf_filter_leye = [AlphaBetaFilter(self.abf_alpha_eye, self.abf_beta_eye, initial_state=self.abf_initial_state,
                                                initial_velocity=self.abf_initial_velocity, facial_model_points=self.facial_model_points)
                                for _ in range(eye_salient_num)]

        # initialize an alpha-beta filter for each of the eye_salient_num reye points
        self.abf_filter_reye = [AlphaBetaFilter(self.abf_alpha_eye, self.abf_beta_eye, initial_state=self.abf_initial_state,
                                                initial_velocity=self.abf_initial_velocity, facial_model_points=self.facial_model_points)
                                for _ in range(eye_salient_num)]

        # initialize an alpha-beta filter for each of the eye_salient_num head_pose_p0 point
        self.abf_filter_head_pose_p0 = AlphaBetaFilter(self.abf_alpha_pose, self.abf_beta_pose, initial_state=self.abf_initial_state,
                                                       initial_velocity=self.abf_initial_velocity, facial_model_points=self.facial_model_points)

        # initialize an alpha-beta filter for each of the eye_salient_num head_pose_p1 point
        self.abf_filter_head_pose_p1 = AlphaBetaFilter(self.abf_alpha_pose, self.abf_beta_pose, initial_state=self.abf_initial_state,
                                                       initial_velocity=self.abf_initial_velocity, facial_model_points=self.facial_model_points)

    def estimate_head_pose(self, head_salient_points, imshape):
        # Salient points
        image_points = np.array([
            # Nose tip: This should correspond to point 30 in dlib's 68-point model.
            head_salient_points[30],
            # Chin: This should correspond to point 8.
            head_salient_points[8],
            # Left eye left corner: This should correspond to point 36.
            head_salient_points[36],
            # Right eye right corner: This should correspond to point 45.
            head_salient_points[45],
            # Left Mouth corner: This should correspond to point 48.
            head_salient_points[48],
            # Right mouth corner: This should correspond to point 54.
            head_salient_points[54]
        ], dtype="double")

        # Camera internals
        focal_length = imshape[1]
        center = (imshape[1]/2, imshape[0]/2)
        camera_matrix = np.array(
            [[focal_length, 0, center[0]],
                [0, focal_length, center[1]],
                [0, 0, 1]], dtype="double"
        )

        dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion
        (success, rotation_vector, translation_vector) = cv2.solvePnP(
            self.facial_model_points, image_points, camera_matrix, dist_coeffs)

        # Project a 3D point (0, 0, 1000.0) onto the image plane.
        # We use this to draw a line sticking out of the nose

        (nose_end_point2D, jacobian) = cv2.projectPoints(np.array(
            [(0.0, 0.0, 1000.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)

        p0 = (int(image_points[0][0]), int(image_points[0][1]))
        p1 = (int(nose_end_point2D[0][0][0]),
              int(nose_end_point2D[0][0][1]))

        return p0, p1

    def measure_brightness(self, img):
        if len(img.shape) == 3:
            # Colored RGB or BGR (*Do Not* use HSV images with this function)
            # create brightness with euclidean norm
            return np.average(norm(img, axis=2)) / np.sqrt(3)
        else:
            # Grayscale
            return np.average(img)

    def create_leye_gaze_model(self, eye_centers, calibration_points):
        """
        Train an RBFInterpolator model using eye center positions and calibration point positions.

        Parameters:
        eye_centers: numpy array of shape (n_samples, 2) with eye center coordinates
        calibration_points: numpy array of shape (n_samples, 2) with calibration point coordinates

        Returns:
        rbfi: trained RBFInterpolator instance
        """

        # Assert that eye_centers and calibration_points have the same size
        assert eye_centers.shape == calibration_points.shape

        # Train the model
        self.leye_gaze_model = RBFInterpolator(eye_centers, calibration_points)

    def create_reye_gaze_model(self, eye_centers, calibration_points):
        assert eye_centers.shape == calibration_points.shape
        self.reye_gaze_model = RBFInterpolator(eye_centers, calibration_points)

    def leye_is_calibrated(self):
        return self.leye_gaze_model != None

    def reye_is_calibrated(self):
        return self.reye_gaze_model != None

    def predict_leye_gaze(self, eye_center):
        # Assert that model exists
        assert self.leye_gaze_model != None

        return self.leye_gaze_model.predict(eye_center)

    def predict_reye_gaze(self, eye_center):
        # Assert that model exists
        assert self.leye_gaze_model != None

        return self.leye_gaze_model.predict(eye_center)

    def predict_pixel_location(self):
        if self.leye_is_calibrated() == False and self.reye_is_calibrated() == False:
            return [0, 0]

    def get_leye_roi(self, shape):
        tl = (shape[42][0], shape[43][1])
        br = (shape[45][0], shape[46][1])
        leye_roi = dlib.rectangle(left=int(tl[0]), top=int(
            tl[1]), right=int(br[0]), bottom=int(br[1]))
        return leye_roi

    def get_reye_roi(self, shape):
        tl = (shape[36][0], shape[37][1])
        br = (shape[39][0], shape[40][1])
        reye_roi = dlib.rectangle(left=int(tl[0]), top=int(
            tl[1]), right=int(br[0]), bottom=int(br[1]))
        return reye_roi

    def define_eye(self, shape):
        centroid = [0, 0]
        pupil_points = []
        iris_points = []

        for i in range(eye_salient_num):
            if i == 0 or i == 1 or (8 <= i < 14):
                centroid[0] += shape[i][0]
                centroid[1] += shape[i][1]
                pupil_points.append((shape[i][0], shape[i][1]))
            else:
                iris_points.append((shape[i][0], shape[i][1]))

        centroid[0] /= 8
        centroid[1] /= 8

        return centroid, pupil_points, iris_points

    def calculate_eye_aspect_ratio(self, eye):
        p2_minus_p6 = dist.euclidean(eye[1], eye[5])
        p3_minus_p5 = dist.euclidean(eye[2], eye[4])
        p1_minus_p4 = dist.euclidean(eye[0], eye[3])
        ear = (p2_minus_p6 + p3_minus_p5) / (2.0 * p1_minus_p4)
        return ear

    def calculate_eyebrow_eye_distance(self, landmarks):
        # Calculate the interocular distance
        left_eye_center = np.mean(landmarks[36:42], axis=0)
        right_eye_center = np.mean(landmarks[42:48], axis=0)
        interocular_distance = dist.euclidean(
            left_eye_center, right_eye_center)

        # These indices are based on the dlib 68-point model
        left_eyebrow_points = landmarks[17:22]
        right_eyebrow_points = landmarks[22:27]
        left_eye_points = landmarks[36:42]
        right_eye_points = landmarks[42:48]

        # Calculate the average y-coordinate of the eyebrow points and eye points
        left_eyebrow_avg_y = np.mean(left_eyebrow_points[:, 1])
        right_eyebrow_avg_y = np.mean(right_eyebrow_points[:, 1])
        left_eye_avg_y = np.mean(left_eye_points[:, 1])
        right_eye_avg_y = np.mean(right_eye_points[:, 1])

        # The eyebrow-eye distance is the difference between the average y-coordinates
        left_eyebrow_eye_distance = left_eye_avg_y - left_eyebrow_avg_y
        right_eyebrow_eye_distance = right_eye_avg_y - right_eyebrow_avg_y

        # Normalize the eyebrow-eye distances by the interocular distance
        left_eyebrow_eye_distance /= interocular_distance
        right_eyebrow_eye_distance /= interocular_distance

        return left_eyebrow_eye_distance, right_eyebrow_eye_distance

    def is_portrait(self, img):
        return img.shape[0] > img.shape[1]

    async def recv(self):
        frame = await self.track.recv()
        try:
            img = frame.to_ndarray(format="bgr24")
            img_draw = img.copy()
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img_small = cv2.resize(
                img, (resize_in_to_width, resize_in_to_height))
            img_gray_small = cv2.cvtColor(img_small, cv2.COLOR_BGR2GRAY)
            is_portrait = self.is_portrait(img_gray_small)

            # width ratio
            scale_factor = img.shape[1] / img_small.shape[1]

            # Ask the face_detector to find the bounding boxes of each face.
            if self.frame_index % 30 == 0:
                # Measure brighness
                self.frame_brightness = self.measure_brightness(img_small)
                # Detect faces
                detected_faces = face_detector(img_gray_small, 0)
                if use_cuda:
                    detected_faces = [[
                        d.rect.left(),
                        d.rect.top(),
                        d.rect.right(),
                        d.rect.bottom()
                    ] for d in detected_faces]
                else:
                    detected_faces = [[
                        d.left(),
                        d.top(),
                        d.right(),
                        d.bottom()
                    ] for d in detected_faces]

                if len(detected_faces) > 0:
                    self.face_fails = 0
                    self.face_rects = detected_faces[0:1]
                else:
                    self.face_fails += 1

            if self.face_fails >= self.face_fails_max:
                self.face_rects = None

            if self.face_rects != None:
                # go through the face bounding boxes
                for rect in self.face_rects:
                    # extract the coordinates of the bounding box
                    x1 = int(rect[0] * scale_factor)
                    y1 = int(rect[1] * scale_factor)
                    x2 = int(rect[2] * scale_factor)
                    y2 = int(rect[3] * scale_factor)

                    # draw it
                    if self.draw:
                        cv2.rectangle(img_draw, (x1, y1),
                                      (x2, y2), (255, 255, 255), 1)

                    # narrow down original area
                    face_img_gray = img_gray[y1:y2, x1:x2]

                    # check if legit
                    if face_img_gray.size != 0:

                        # calculate aspect ratio
                        aspect_ratio = max(1, face_img_gray.shape[1]) / \
                            max(1, face_img_gray.shape[0])

                        # calculate new dimensions
                        desired_height = 400
                        desired_width = int(desired_height * aspect_ratio)

                        # resize the face region while preserving the aspect ratio
                        face_img_gray_resized = cv2.resize(
                            face_img_gray, (desired_width, desired_height))

                        # calculate the scale factors
                        scale_factor_x = face_img_gray_resized.shape[1] / \
                            max(1, face_img_gray.shape[1])
                        scale_factor_y = face_img_gray_resized.shape[0] / \
                            max(1, face_img_gray.shape[0])

                        scaled_rect = dlib.rectangle(
                            left=0, top=0, right=face_img_gray_resized.shape[1],
                            bottom=face_img_gray_resized.shape[0])

                        # apply the shape face_predictor to the face ROI
                        face_shape = self.face_predictor(
                            face_img_gray_resized, scaled_rect)

                        # Smooth
                        smoothed_face_points = []
                        for n in range(0, face_salient_num):
                            smoothed_face_x, smoothed_face_y = self.abf_filter_face[n].update(
                                (face_shape.part(n).x, face_shape.part(n).y))
                            smoothed_face_points.append(
                                (smoothed_face_x, smoothed_face_y))

                        # Convert data for face pose
                        smoothed_face_points_np = np.array(
                            smoothed_face_points, dtype="double")

                        # Draw salient
                        for n in range(0, face_salient_num):
                            x = x1 + \
                                int(smoothed_face_points[n]
                                    [0] / scale_factor_x)
                            y = y1 + \
                                int(smoothed_face_points[n]
                                    [1] / scale_factor_y)
                            if self.draw:
                                cv2.circle(img_draw, (int(x), int(y)),
                                           1, (255, 255, 255), 1)

                        # Estimate pose
                        p0, p1 = self.estimate_head_pose(
                            smoothed_face_points_np, img_gray.shape)

                        smoothed_p0_x, smoothed_p0_y = self.abf_filter_head_pose_p0.update(
                            (p0[0], p0[1]))

                        smoothed_p1_x, smoothed_p1_y = self.abf_filter_head_pose_p1.update(
                            (p1[0], p1[1]))

                        # Rescale
                        p0 = (x1+int(smoothed_p0_x / scale_factor_x),
                              y1+int(smoothed_p0_y / scale_factor_y))

                        p1 = (x1+int(smoothed_p1_x / scale_factor_x),
                              y1+int(smoothed_p1_y / scale_factor_y))

                        # Draw pose
                        if self.draw:
                            cv2.line(img_draw, p0, p1, (255, 255, 0), 1)

                        # Eyebrow movement detection
                        left_eyebrow_eye_distance, right_eyebrow_eye_distance = self.calculate_eyebrow_eye_distance(
                            smoothed_face_points_np)

                        # if left_eyebrow_eye_distance > self.leye_eyebrow_treshold:
                        # logger.info(left_eyebrow_eye_distance)
                        # logger.info(self.frame_brightness)

                        # if right_eyebrow_eye_distance > self.reye_eyebrow_treshold:
                        # logger.info(right_eyebrow_eye_distance)

                        # Blink detection
                        (leye_idx_start,
                         leye_idx_end) = self.facial_landmarks["left_eye"]
                        (reye_idx_start,
                         reye_idx_end) = self.facial_landmarks["right_eye"]

                        leye_region = smoothed_face_points_np[leye_idx_start:leye_idx_end]
                        reye_region = smoothed_face_points_np[reye_idx_start:reye_idx_end]

                        leye_ear = self.calculate_eye_aspect_ratio(leye_region)
                        reye_ear = self.calculate_eye_aspect_ratio(reye_region)

                        # RIGHT EYE
                        if reye_ear > self.reye_ear_treshold:

                            # Landmark detection (right eye)
                            reye_rect = self.get_reye_roi(smoothed_face_points)

                            # create a dlib rectangle object for the right eye region of interest (ROI)
                            reye_r = dlib.rectangle(left=reye_rect.left(), top=reye_rect.top(),
                                                    right=reye_rect.right(), bottom=reye_rect.bottom())

                            # detect landmarks in the right eye ROI
                            reye_shape = self.reye_pose_model(
                                face_img_gray_resized, reye_r)

                            # Smooth
                            smoothed_reye_points = []
                            for n in range(0, eye_salient_num):
                                smoothed_reye_x, smoothed_reye_y = self.abf_filter_reye[n].update(
                                    (reye_shape.part(n).x, reye_shape.part(n).y))
                                smoothed_reye_points.append(
                                    (smoothed_reye_x, smoothed_reye_y))

                            # draw the eye on the image
                            reye_center, reye_pupil_points, reye_iris_points = self.define_eye(
                                smoothed_reye_points)

                            # calculate bottom point of the eye
                            # reye_btm = ((smoothed_face_points[36][0] + smoothed_face_points[39][0]) / 2, (smoothed_face_points[36][1] + smoothed_face_points[39][1]) / 2)

                            # compute the angle from the bottom of the eye to the center of the pupil
                            # reye_angle = math.atan2(reye_btm[1] - reye_center[1], reye_btm[0] - reye_center[0]) * 180 / math.pi

                            # compute the radius of the eye
                            # reye_radius = np.linalg.norm(np.array([smoothed_face_points[36][0], smoothed_face_points[36][1]]) - np.array([smoothed_face_points[39][0], smoothed_face_points[39][1]]))

                            # compute the distance from the bottom of the eye to the center of the pupil
                            # reye_pupil_center = np.linalg.norm(np.array(reye_btm) - np.array([reye_center[0], reye_center[1]]))

                            # Scale back the eye center and other points
                            reye_center = (int(reye_center[0] / scale_factor_x),
                                           int(reye_center[1] / scale_factor_y))
                            # reye_pupil_points = [(int(x[0] / scale_factor_x), int(x[1] / scale_factor_y)) for x in reye_pupil_points]
                            # reye_iris_points = [(int(x[0] / scale_factor_x), int(x[1] / scale_factor_y)) for x in reye_iris_points]

                            # convert our reye_pupil_points and reye_iris_points to numpy arrays of type int32 and reshape them accordingly
                            # reye_pupil_points_np = np.array(reye_pupil_points, dtype=np.int32).reshape((-1, 1, 2))
                            # reye_iris_points_np = np.array(reye_iris_points, dtype=np.int32).reshape((-1, 1, 2))

                            # Draw it
                            if self.draw:
                                cv2.circle(img_draw, (int(
                                    x1 + reye_center[0]), int(y1 + reye_center[1])), 1, (0, 255, 0), thickness=-1)
                                # cv2.polylines(img_draw, [reye_pupil_points_np + [x1, y1]], isClosed=True, color=(255, 255, 0), thickness=1)
                                # cv2.polylines(img_draw, [reye_iris_points_np + [x1, y1]], isClosed=True, color=(0, 255, 255), thickness=1)

                            # END OF right eye.

                        # LEFT EYE
                        if leye_ear > self.leye_ear_treshold:

                            # Landmark detection (left eye)
                            leye_rect = self.get_leye_roi(smoothed_face_points)

                            # create a dlib rectangle object for the left eye region of interest (ROI)
                            leye_r = dlib.rectangle(left=leye_rect.left(), top=leye_rect.top(),
                                                    right=leye_rect.right(), bottom=leye_rect.bottom())

                            # detect landmarks in the left eye ROI
                            leye_shape = self.leye_pose_model(
                                face_img_gray_resized, leye_r)

                            # Smooth
                            smoothed_leye_points = []
                            for n in range(0, eye_salient_num):
                                smoothed_leye_x, smoothed_leye_y = self.abf_filter_leye[n].update(
                                    (leye_shape.part(n).x, leye_shape.part(n).y))
                                smoothed_leye_points.append(
                                    (smoothed_leye_x, smoothed_leye_y))

                            # draw the eye on the image
                            leye_center, leye_pupil_points, leye_iris_points = self.define_eye(
                                smoothed_leye_points)

                            # calculate bottom point of the eye
                            # leye_btm = ((smoothed_face_points[45][0] + smoothed_face_points[42][0]) / 2, (smoothed_face_points[45][1] + smoothed_face_points[42][1]) / 2)

                            # compute the angle from the bottom of the eye to the center of the pupil
                            # leye_angle = math.atan2(leye_btm[1] - leye_center[1], leye_btm[0] - leye_center[0]) * 180 / math.pi

                            # compute the radius of the eye
                            # leye_radius = np.linalg.norm(np.array([smoothed_face_points[42][0], smoothed_face_points[42][1]]) - np.array([smoothed_face_points[45][0], smoothed_face_points[45][1]]))

                            # compute the distance from the bottom of the eye to the center of the pupil
                            # leye_pupil_center = np.linalg.norm(np.array(leye_btm) - np.array([leye_center[0], leye_center[1]]))

                            # Scale back the eye center and other points
                            leye_center = (int(leye_center[0] / scale_factor_x),
                                           int(leye_center[1] / scale_factor_y))
                            # leye_pupil_points = [(int(x[0] / scale_factor_x), int(x[1] / scale_factor_y)) for x in leye_pupil_points]
                            # leye_iris_points = [(int(x[0] / scale_factor_x), int(x[1] / scale_factor_y)) for x in leye_iris_points]

                            # convert our leye_pupil_points and leye_iris_points to numpy arrays of type int32 and reshape them accordingly
                            # leye_pupil_points_np = np.array(leye_pupil_points, dtype=np.int32).reshape((-1, 1, 2))

                            # leye_iris_points_np = np.array(leye_iris_points, dtype=np.int32).reshape((-1, 1, 2))

                            # draw it
                            if self.draw:
                                cv2.circle(img_draw, (int(
                                    x1 + leye_center[0]), int(y1 + leye_center[1])), 1, (0, 255, 0), thickness=-1)
                                # cv2.polylines(img_draw, [leye_pupil_points_np + [x1, y1]], isClosed=True, color=(255, 255, 0), thickness=1)
                                # cv2.polylines(img_draw, [leye_iris_points_np + [x1, y1]], isClosed=True, color=(0, 255, 255), thickness=1)
                            # END OF right eye.

            # rebuild a VideoFrame, preserving timing information
            # keep small format
            if is_portrait:
                img_draw = cv2.resize(
                    img_draw, (resize_out_to_height, resize_out_to_width))
            else:
                img_draw = cv2.resize(
                    img_draw, (resize_out_to_width, resize_out_to_height))

            new_frame = VideoFrame.from_ndarray(img_draw, format="bgr24")
            new_frame.pts = frame.pts
            new_frame.time_base = frame.time_base
            # increment frame index
            self.frame_index = self.frame_index + 1
            # return
            return new_frame

        except Exception as e:
            logger.error(f"Error processing frame: {e}")
            return frame


async def signal(request):
    params = await request.json()
    signal = RTCSessionDescription(sdp=params["sdp"], type=params["type"])
    session_engine = None

    pc = RTCPeerConnection()
    pcs.add(pc)

    def maybe_send_message(channel, message):
        if pc.connectionState == "connected":
            try:
                channel.send(message)
            except:
                logger.error("Message send fail", )

    @pc.on("datachannel")
    def on_datachannel(channel):

        @channel.on("message")
        def on_message(message):

            if isinstance(message, str):

                if message.startswith("ping"):
                    maybe_send_message(
                        channel=channel, message="pong" + message[4:])

                elif message.startswith("xy"):
                    if session_engine == None:
                        maybe_send_message(channel=channel, message="[0]")
                    elif session_engine.is_calibrated():
                        maybe_send_message(channel=channel, message=json.dumps(
                            session_engine.predict_pixel_location()))
                    else:
                        maybe_send_message(channel=channel, message="[1]")

                elif message.startswith("cb"):
                    if session_engine == None:
                        maybe_send_message(channel=channel, message="[0]")
                    else:
                        calibration_data = json.loads(message[2:])
                        leye_centers = np.array(calibration_data[0])
                        reye_centers = np.array(calibration_data[1])
                        calibration_points = np.array(calibration_data[2])
                        session_engine.create_leye_gaze_model(
                            leye_centers, calibration_points)
                        session_engine.create_reye_gaze_model(
                            reye_centers, calibration_points)
                        maybe_send_message(channel=channel, message=json.dumps(
                            session_engine.predict_pixel_location()))

    @ pc.on("track")
    def on_track(track):
        logger.info("Track %s received", track.kind)
        if track.kind == "video":
            session_engine = EyeTrackingEngine(
                track=relay.subscribe(track),
                face_detector=face_detector,
                face_predictor=face_predictor,
                facial_landmarks=facial_landmarks,
                facial_model_points=facial_model_points
            )
            pc.addTrack(session_engine)

        @ track.on("ended")
        async def on_ended():
            logger.info("Track %s ended", track.kind)

    @ pc.on("connectionstatechange")
    async def on_connectionstatechange():
        print("Connection state is %s" % pc.connectionState)
        if pc.connectionState == "failed":
            await pc.close()
            pcs.discard(pc)

    await pc.setRemoteDescription(signal)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return web.Response(
        content_type="application/json",
        text=json.dumps(
            {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}
        ),
    )


async def on_shutdown(app):
    # close peer connections
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros)
    pcs.clear()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gaze Explorer 0.2.0 (beta)")
    parser.add_argument(
        "--cert-file", help="SSL certificate file (for HTTPS)")
    parser.add_argument("--key-file", help="SSL key file (for HTTPS)")
    parser.add_argument("--host", default="localhost",
                        help="Host for HTTP server (default: localhost)")
    parser.add_argument("--port", type=int, default=9090,
                        help="Port for HTTP server (default: 9090)")
    parser.add_argument("--verbose", "-v", action="count", default="")

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    if args.cert_file:
        ssl_context = ssl.SSLContext()
        ssl_context.load_cert_chain(args.cert_file, args.key_file)
    else:
        ssl_context = None

    app = web.Application()
    app.on_shutdown.append(on_shutdown)
    app.router.add_post("/signal", signal)
    cors = aiohttp_cors.setup(app, defaults={
        "*": aiohttp_cors.ResourceOptions(
            allow_credentials=True,
            expose_headers="*",
            allow_headers="*"
        )
    })

    for route in list(app.router.routes()):
        cors.add(route)

    web.run_app(app, host=args.host, port=args.port,
                ssl_context=ssl_context)
