FROM nvcr.io/nvidia/tritonserver:21.05-py3

ENV TZ=Asia/Almaty
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
RUN export LC_ALL=C.UTF-8
RUN export LANG=C.UTF-8

RUN DEBIAN_FRONTEND="noninteractive" apt-get update && apt-get -y install tzdata sudo

RUN apt-get update \
    && apt-get install -y \
    python \
    python3-dev \
    python3-pip \
  && apt-get clean

RUN curl https://bootstrap.pypa.io/pip/get-pip.py -o get-pip.py && python3.8 get-pip.py --force-reinstall
RUN sudo pip3 install transformers flask fastapi python-multipart numpy requests
RUN sudo pip3 install uvicorn
RUN sudo pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
RUN sudo pip3 install keybert
RUN sudo pip3 install PyPDF2
RUN sudo pip3 install beautifulsoup4
RUN sudo pip3 install openai


ADD  src/ /opt/api/
WORKDIR /opt/api/
ENV PYTHONUNBUFFERED=1

ENTRYPOINT ["sh", "entrypoint.sh"]
