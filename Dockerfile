FROM vllm/vllm-openai:v0.18.0

COPY ./turboquant /opt/turboquant
WORKDIR /opt/turboquant

RUN pip config --user set global.index https://mirrors.tools.huawei.com/pypi && \
    pip config --user set global.index-url https://mirrors.tools.huawei.com/pypi/simple && \
    pip config --user set global.trusted-host mirrors.tools.huawei.com

RUN pip install -e .

COPY start_with_turboquant.py /app/start_with_turboquant.py
RUN ln -s /opt/turboquant /app/turboquant

WORKDIR /app

ENTRYPOINT ["python3", "/app/start_with_turboquant.py"]
