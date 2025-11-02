ARG BASE_IMAGE=lmsysorg/sglang:v0.5.4.post1
FROM ${BASE_IMAGE}

COPY serve /usr/bin/serve
RUN chmod 777 /usr/bin/serve

# Set the entrypoint to our serve script
ENTRYPOINT [ "/usr/bin/serve" ]