# Head motion server
This is server part of head motion recognition system
# Requirements
1. Python 3.7
2. OpenCV
3. Tensorflow 2.1
4. gRPC
# Setup
1. Clone this repository
```bash
https://github.com/rivalocop/grpcserver.git
```
2. Setup environment and install neccessary packages
```bash
cd grpcserver
python3 -m venv .env
source .env/bin/activate # activate the environment
pip install -r requirements.txt # install packages
```
3. Run the server
```bash
python server.py
```
