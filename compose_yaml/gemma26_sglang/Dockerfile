FROM dgx-sglang-gemma4-31b-dflash:2ef0fe4-toolindex1
COPY chat_template.jinja patch_tool_index.py patch_rope.py launch.py /opt/gemma4/
RUN python3 /opt/gemma4/patch_tool_index.py && python3 /opt/gemma4/patch_rope.py
ENTRYPOINT ["python3", "/opt/gemma4/launch.py"]
