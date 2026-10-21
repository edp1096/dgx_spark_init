FROM lmsysorg/sglang@sha256:12d3392bdc8be8d35e9a95f191df6aef99c5114bdbefd41bfdc7e760e6d25ec1
ARG TENSORFOLD_REVISION=db281878ddb836fd0df510d8771ecb7e0fe47d26
RUN git init /opt/tensorfold && git -C /opt/tensorfold remote add origin https://github.com/ashhart/TensorFold.git \
 && git -C /opt/tensorfold fetch --depth 1 origin ${TENSORFOLD_REVISION} \
 && git -C /opt/tensorfold checkout --detach FETCH_HEAD \
 && test "$(git -C /opt/tensorfold rev-parse HEAD)" = "${TENSORFOLD_REVISION}"
COPY launch.py prepare_vision.py compact_experts.py parallel_group.py /opt/sparktalk-qwen-exl3/
ENV PYTHONPATH=/opt/tensorfold/src PYTHONUNBUFFERED=1 MAX_JOBS=1 TORCH_CUDA_ARCH_LIST=12.1
ENTRYPOINT ["python3", "-u", "/opt/sparktalk-qwen-exl3/launch.py"]
