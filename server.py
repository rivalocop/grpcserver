import io
import logging
import grpc
import motion_pb2
import motion_pb2_grpc
import numpy as np
import cv2
import settings
from face_detector.face_detect import ImFace
from concurrent import futures
from PIL import Image


print("[INFO] loading face detector...")
net = cv2.dnn.readNetFromCaffe(settings.DEPLOY_FILE, settings.CAFFE_MODEL)
print("[INFO] loading face detector done...")


class MotionServicer(motion_pb2_grpc.MotionServicer):
    def MotionStreaming(self, request_iterator, context):
        for ri in request_iterator:
            with io.BytesIO(ri.frame) as f:
                image = Image.open(f)
                image_as_array = np.asarray(image)
                extracted_face = ImFace(image_as_array, net).face

                # image = np.frombuffer(ri.frame, dtype='uint8')
                # image = image.reshape(ri.width, ri.height)
                # print(image)
                yield motion_pb2.MotionResponse(result=True)


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
