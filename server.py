import logging
import pickle
from concurrent import futures
from datetime import datetime
from time import time
from typing import List, Optional

import cv2
import face_recognition
import grpc
import numpy as np
import tensorflow as tf
from bson.objectid import ObjectId
from pydantic.main import BaseModel

import motion_pb2
import motion_pb2_grpc
import settings
from face_detector.face_detect import ImFace
from motion_detector.motion_detect import ImMotion
from repository.singleton import faces, users, activities
import time

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices('GPU')
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


class RecentActivity(BaseModel):
    isSuccess: bool = False
    title: str
    content: str
    causeId: str
    createdTime: datetime


def preprocess_image(image):
    img = np.frombuffer(image, np.uint8).reshape(352, 288, 4)
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    img = cv2.resize(img, (300, 300))
    img = cv2.flip(img, 1)
    return img


def get_face_indexes(user_id):
    face_indexes = faces.find({'userId': user_id})
    return [doc["encoding"] for doc in face_indexes]


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
    except Exception as err:
        print(err)


class MotionServicer(motion_pb2_grpc.MotionServicer):
    def MotionStreaming(self, request_iterator, context):
        for ri in request_iterator:
            if ri.isPingMsg:
                print("Pinged")
                yield motion_pb2.MotionResponse(isPongMsg=True)
            else:
                returned_model = motion_pb2.MotionResponse(
                    result=False, confidence=0.0, id='None')
                try:
                    image = preprocess_image(ri.imagePayload)
                    box = face_recognition.face_locations(image)
                    if len(box) > 0:
                        result = predict_process(image, box)
                        print(
                            f"predict {ri.expectedLabel} got {result.label} with confidence: {result.confidence}")
                        if result.label == ri.expectedLabel and result.confidence > 0.5:
                            encoding_id = faces.insert_one(
                                result.dict()).inserted_id
                            returned_model = motion_pb2.MotionResponse(
                                result=True, confidence=result.confidence, id=str(encoding_id))
                except Exception as e:
                    print(e)
                finally:
                    time.sleep(0.5)
                    print('predicted')
                    yield returned_model

    def RegisterFaceIndexes(self, request, context):
        registered_user = User(userId=request.userId)
        users.insert_one(registered_user.dict())
        return motion_pb2.UserFormData(userId=registered_user.userId)

    def UpdateFaceIndexes(self, request, context):
        returned_model = motion_pb2.FaceIndexesResponse(isSuccess=False)
        try:
            if len(request.imageIds) > 0:
                for i in request.imageIds:
                    faces.update_one({'_id': ObjectId(i)}, {
                                     '$set': {'userId': request.userInf.userId}})

                new_activity = RecentActivity(
                    isSuccess=True,
                    title='Update face indexes',
                    content='Update face indexes from main account',
                    causeId=request.userInf.userId,
                    createdTime=datetime.now()
                )
                activity_id = activities.insert_one(
                    new_activity.dict()).inserted_id
                returned_model = motion_pb2.FaceIndexesResponse(
                    isSuccess=new_activity.isSuccess, activityId=str(activity_id))
        except Exception as err:
            print(err)
        finally:
            return returned_model

    def FaceRecognizeStreaming(self, request_iterator, context):
        face_embedding = []
        for ri in request_iterator:
            if ri.isPingMsg:
                print("Pinged")
                face_embedding = get_face_indexes(ri.userInf.userId)
                yield motion_pb2.MotionResponse(isPongMsg=True)
            else:
                returned_model = motion_pb2.MotionResponse(
                    result=False, confidence=0.0, id='None')
                try:
                    image = preprocess_image(ri.imagePayload)
                    box = face_recognition.face_locations(image)
                    if len(box) > 0:
                        result = predict_process(image, box)
                        print(
                            f"predict {ri.expectedLabel} got {result.label} with confidence: {result.confidence}")
                        if result.label == ri.expectedLabel and result.confidence > 0.5:
                            unknown_face_encoding = np.array(result.encoding)
                            match_results = face_recognition.compare_faces(face_embedding, unknown_face_encoding)
                            if match_results[0]:
                                returned_model = motion_pb2.MotionResponse(
                                    result=True)
                except Exception as err:
                    print(err)
                finally:
                    yield returned_model


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
