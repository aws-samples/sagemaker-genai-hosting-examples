FROM ghcr.io/predibase/lorax:0.7.0

RUN apt-get install wget
RUN wget \
    https://raw.githubusercontent.com/predibase/lorax/v0.8.1/server/lorax_server/utils/sources/__init__.py \
    https://raw.githubusercontent.com/predibase/lorax/v0.8.1/server/lorax_server/utils/sources/s3.py \
    && mv -t /usr/src/server/lorax_server/utils/sources/ __init__.py s3.py

COPY sagemaker_entrypoint.sh entrypoint.sh
RUN chmod +x entrypoint.sh

ENTRYPOINT ["./entrypoint.sh"]
