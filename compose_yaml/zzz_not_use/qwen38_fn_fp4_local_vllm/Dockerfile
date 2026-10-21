FROM eugr/spark-vllm-b12x@sha256:03a956544e9f9b4b7ba91690469fbe7848bdbef29a7d2e0f5788368467bb3595

COPY fix_yarn_config.py /opt/fp4-local/fix_yarn_config.py
RUN python3 /opt/fp4-local/fix_yarn_config.py

LABEL sparktalk.fp4-local.yarn-config="2" \
      sparktalk.fp4-local.base-digest="sha256:03a956544e9f9b4b7ba91690469fbe7848bdbef29a7d2e0f5788368467bb3595"
