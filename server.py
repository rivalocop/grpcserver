import io
import logging
import pickle

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


class MotionServicer(motion_pb2_grpc.MotionServicer):
    def MotionStreaming(self, request_iterator, context):
        for ri in request_iterator:
            with io.BytesIO(ri.frame) as f:
                image = Image.open(f)
                image_as_array = np.asarray(image)
                extracted_face = ImFace(image_as_array, net).face
                motion = ImMotion(extracted_face, le, model, ri.expectedLabel)
                yield motion_pb2.MotionResponse(result=motion.result)


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
