import logging
from concurrent import futures

import grpc

import motion_pb2_grpc
from services.motion import MotionServicer


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
