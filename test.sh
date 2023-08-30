# -t flag tags your image.
docker build -t sagecontinuum/sage-smoke-detection:0.1.0 .

docker run -v ${PWD}/pywaggle-logs:/src/pywaggle-logs \
    --rm -it --entrypoint bash sagecontinuum/sage-smoke-detection:0.1.0

docker run -it sagecontinuum/sage-smoke-detection:0.1.0 bash

docker container exec -it  sagecontinuum/sage-smoke-detection:0.1.0 bash

docker run  \
    -v ${PWD}/pywaggle-logs:/src/pywaggle-logs \
    --env PYWAGGLE_LOG_DIR=pywaggle-logs sagecontinuum/sage-smoke-detection:0.1.0 \
    -c scr/20190610-Pauma-bh-w-mobo-c.mp4

docker run  \
    -v ${PWD}/pywaggle-logs:/src/pywaggle-logs \
    -v src:/src_git \
    --env PYWAGGLE_LOG_DIR=pywaggle-logs \
    --entrypoint bash \
    sagecontinuum/sage-smoke-detection:0.1.0

docker run -v ${PWD}/pywaggle-logs:/src/pywaggle-logs -v ${PWD}/src/:/src_git \
    --env PYWAGGLE_LOG_DIR=pywaggle-logs \
    --rm -it --entrypoint bash sagecontinuum/sage-smoke-detection:0.1.0
