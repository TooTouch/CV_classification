nvidia-docker run -it -h cv_practice \
        -p 1290:1290 \
        --ipc=host \
        --name cv_practice \
        -v /m2:/projects \
        nvcr.io/nvidia/pytorch:22.12-py3 bash
