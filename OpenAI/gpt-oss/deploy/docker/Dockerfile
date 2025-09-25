FROM vllm/vllm-openai:v0.10.2

COPY serve /usr/bin/serve
RUN chmod 777 /usr/bin/serve

ENTRYPOINT [ "/usr/bin/serve" ]
