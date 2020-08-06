import io
import logging
import pickle
from concurrent import futures
from typing import List, Optional

import cv2
import face_recognition
import grpc
import numpy as np
import tensorflow as tf
from PIL import Image
from pydantic.main import BaseModel

import motion_pb2
import motion_pb2_grpc
import settings
from face_detector.face_detect import ImFace
from motion_detector.motion_detect import ImMotion
from repository.singleton import faces, users

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


class FaceIndexes(BaseModel):
    userId: Optional[str] = None
    confidence: float
    label: str
    encoding: List[float]


class User(BaseModel):
    userId: str


def preprocess_image(image):
    with io.BytesIO(image) as f:
        # Load it as numpy array
        image = np.array(Image.open(f))
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image = cv2.resize(image, (300, 300))
        return image


def get_face_indexes(user_id):
    face_embeddings = []
    face_indexes = faces.find_one({'userId': user_id})
    if len(face_indexes['encodings']) != 0:
        return face_indexes['encodings']
    return face_embeddings


def predict_process(image, location):
    try:
        extracted_face = ImFace(image, net).face
        motion = ImMotion(extracted_face, le,
                          model)
        encodings = face_recognition.face_encodings(image, location)
        new_embedding = FaceIndexes(
            confidence=motion.result['confidence'],
            label=motion.result['label'],
            encoding=encodings[0].tolist())
        return new_embedding
    except Exception as e:
        print(e)


class MotionServicer(motion_pb2_grpc.MotionServicer):
    def MotionStreaming(self, request_iterator, context):
        for ri in request_iterator:
            if ri.isPingMsg:
                print("Pinged")
                yield motion_pb2.MotionResponse(isPongMsg=True)
            else:
                image = preprocess_image(ri.imagePayload)
                box = face_recognition.face_locations(image, model='hog')
                if len(box) > 0:
                    result = predict_process(image, box)
                    print(result.confidence)
                    print(f"predict {ri.expectedLabel} got {result.label}")
                    if result.label == ri.expectedLabel and result.confidence > 0.5:
                        encoding_id = faces.insert_one(
                            result.dict()).inserted_id
                        yield motion_pb2.MotionResponse(result=True, confidence=result.confidence, id=str(encoding_id))
                    else:
                        yield motion_pb2.MotionResponse(result=False, confidence=0.0, id='None')

    def RegisterFaceIndexes(self, request, context):
        registered_user = User(userId=request.userId)
        users.insert_one(registered_user.dict())
        return motion_pb2.UserFormData(userId=registered_user.userId)


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    motion_pb2_grpc.add_MotionServicer_to_server(MotionServicer(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    server.wait_for_termination()


if __name__ == "__main__":
    logging.basicConfig()
    print("[INFO] Starting server...")
    serve()
