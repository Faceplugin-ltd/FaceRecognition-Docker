# Face-Recognition-Linux — recognition-only
# Native libs are linux/amd64; image runs on any Docker host (Win/Mac/Linux).
# Before build: download Drive folder contents into ./lib/cpu/ (see lib/README.md)
# Bookworm is glibc 2.36; native SDK libs need GLIBC_2.38+ (Debian Trixie).
FROM --platform=linux/amd64 python:3.12-slim-trixie

RUN apt-get update -y && apt-get install -y --no-install-recommends \
        psmisc curl util-linux e2fsprogs libgomp1 libuuid1 zlib1g libusb-0.1-4 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /root/facesdk

COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

COPY app.py sdk.py gallery.py quality_gate.py openapi.json license_ux.py http_body.py run.sh ./
COPY lib ./lib/

ENV LICENSE=/root/facesdk/license.txt
ENV LD_LIBRARY_PATH=/root/facesdk/lib/cpu
ENV PORT=8092
# Published port only works if the process listens on every interface.
# FACESDK_DOCKER allows that without an API key. Local ./run.sh stays on 127.0.0.1.
ENV FACESDK_BIND_HOST=0.0.0.0
ENV FACESDK_DOCKER=1
RUN chmod +x ./run.sh \
    && test -f ./lib/cpu/libFaceRecognitionSDK.so \
    && test -f ./lib/cpu/libfacerec.so \
    && test -f ./lib/cpu/far.fpk \
    && test -f ./lib/cpu/libonnxruntime.so \
    && test -f ./lib/cpu/libonnxruntime_providers_shared.so

CMD ["./run.sh"]
EXPOSE 8092
