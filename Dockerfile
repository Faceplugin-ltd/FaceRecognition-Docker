# FaceRecognition-Linux — Linux SDK + Docker (same repo)
# Native libs are linux/amd64; image runs on any Docker host (Win/Mac/Linux).
# Before build: download Drive folder contents into ./lib/cpu/ (see lib/README.md)
# Bookworm is glibc 2.36; native SDK libs need GLIBC_2.38+ (Debian Trixie).
FROM --platform=linux/amd64 python:3.12-slim-trixie

RUN apt-get update -y && apt-get install -y --no-install-recommends \
        psmisc curl util-linux e2fsprogs libgomp1 libuuid1 zlib1g \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /root/facesdk

COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

COPY app.py sdk.py license_ux.py run.sh ./
COPY lib ./lib/

ENV LICENSE=/root/facesdk/license.txt
ENV LD_LIBRARY_PATH=/root/facesdk/lib/cpu
# Listen on product API port 8083 (same as host publish)
ENV PORT=8083
RUN chmod +x ./run.sh \
    && test -f ./lib/cpu/libFaceRecognitionSDK.so \
    && test -f ./lib/cpu/libfar-eng.so \
    && test -f ./lib/cpu/far.fpk

CMD ["./run.sh"]
EXPOSE 8083
