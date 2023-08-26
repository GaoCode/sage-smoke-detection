docker run  \
    -v ${PWD}/pywaggle-logs:/src/pywaggle-logs \
    --env PYWAGGLE_LOG_DIR=pywaggle-logs sagecontinuum/sage-smoke-detection:0.1.0 \
    -c scr/20190610-Pauma-bh-w-mobo-c.mp4