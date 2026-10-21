FROM local/qwen38-fp4-local:csf-yarn-20261016
COPY apply_control_trim.py /opt/vllm-control/apply_control_trim.py
RUN python3 /opt/vllm-control/apply_control_trim.py
ENV VLLM_CONTROL_TRIM_CACHE=1
