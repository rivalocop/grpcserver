import io
import json
import logging
import pickle
import uuid

import face_recognition
import grpc
import motion_pb2
import motion_pb2_grpc
import numpy as np
import cv2
import settings
import tensorflow as tf

from face_detector.face_detect import ImFace
from concurrent import futures
from PIL import Image

from motion_detector.motion_detect import ImMotion
from repository.mongo_repository import MongoRepository
from repository.singleton import faces

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(
            logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

print("[INFO] loading face detector...")
net = cv2.dnn.readNetFromCaffe(settings.DEPLOY_FILE, settings.CAFFE_MODEL)
print("[INFO] loading face detector done...")
print("[INFO] loading motion detector...")
model = tf.keras.models.load_model(settings.MOTION)
le = pickle.loads(open(settings.LABELS, "rb").read())
face_indexes_repo = MongoRepository(faces)


class MotionServicer(motion_pb2_grpc.MotionServicer):
    def RegisterStreaming(self, request_iterator, context):
        encodings = []
        # Register face via streaming frame from camera
        # Loop over stream
        for ri in request_iterator:
            with io.BytesIO(ri.frame.imagePayload) as f:
                final_result = False
                print(ri.frame.expectedLabel)
                # Load it as numpy array
                image = face_recognition.load_image_file(f)
                # Check if image has face or not
                box = face_recognition.face_locations(image)
                if len(box) > 0:
                    # Motion Detection
                    extracted_face = ImFace(image, net).face
                    motion = ImMotion(extracted_face, le,
                                      model, ri.frame.expectedLabel)
                    print(motion.result)
                    if motion.result:
                        # If motion detection passed then extract face encoding for register face indexes
                        encoding = face_recognition.face_encodings(image, box)
                        encodings.append(encoding[0].tolist())
                        final_result = True
                yield motion_pb2.MotionResponse(result=final_result)
        if len(encodings) > 0:
            print("[INFO] serializing encodings...")
            user_id = str(uuid.uuid4())
            data = {"encodings": encodings, "user_id": user_id}
            face_indexes_repo.create(data)

    def AuthenticateStreaming(self, request_iterator, context):
        # Authenticate user via streaming frame from camera
        # Loop over the stream
        known_face_encoding = []
        for ri in request_iterator:
            if len(known_face_encoding) == 0:
                known_face_encoding = face_indexes_repo.get(ri.userId)
                known_face_encoding = np.array(
                    known_face_encoding["encodings"])
            # Open data frame and assign to bytes buffer
            with io.BytesIO(ri.frame.imagePayload) as f:
                final_result = False
                # Load it via face_recognition
                image = face_recognition.load_image_file(f)
                # Check if image has face or not
                box = face_recognition.face_locations(image)
                if len(box) > 0:
                    # Motion detection
                    extracted_face = ImFace(image, net).face
                    motion = ImMotion(extracted_face, le,
                                      model, ri.frame.expectedLabel)
                    if motion.result:
                        # If motion detection passed then extract face encoding
                        encoding = face_recognition.face_encodings(image, box)
                        # Compare it to index stored in mongodb
                        match_results = face_recognition.compare_faces(
                            known_face_encoding, encoding[0])
                        if match_results:
                            final_result = True
                yield motion_pb2.MotionResponse(result=final_result)


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    motion_pb2_grpc.add_MotionServicer_to_server(MotionServicer(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    server.wait_for_termination()


if __name__ == "__main__":
    logging.basicConfig()
    print("[INFO] Starting server...")
    serve()
    print("[INFO] Server Started!")
