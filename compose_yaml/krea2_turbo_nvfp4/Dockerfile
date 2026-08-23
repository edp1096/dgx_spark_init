FROM dgx-flux2-klein-nvfp4:4b

RUN git clone https://github.com/facok/comfyui-krea2-controlnet.git \
      /opt/ComfyUI/custom_nodes/comfyui-krea2-controlnet \
    && cd /opt/ComfyUI/custom_nodes/comfyui-krea2-controlnet \
    && git checkout 79ebfd3bd80d2180b334dd7ce57f3c9ddaa0848f

RUN git clone https://github.com/lbouaraba/comfyui-krea2edit.git \
      /opt/ComfyUI/custom_nodes/comfyui-krea2edit \
    && cd /opt/ComfyUI/custom_nodes/comfyui-krea2edit \
    && git checkout 86f886dac23013d88996e3a2e99093ba44d322fb

RUN git clone https://github.com/ethanfel/ComfyUI-Krea2TextEncoder.git \
      /opt/ComfyUI/custom_nodes/ComfyUI-Krea2TextEncoder \
    && cd /opt/ComfyUI/custom_nodes/ComfyUI-Krea2TextEncoder \
    && git checkout 38da10b0d4655098d867c14af10093baa76a85c4

RUN git clone https://github.com/Nynxz/ComfyUI-NK2E.git \
      /opt/ComfyUI/custom_nodes/ComfyUI-NK2E \
    && cd /opt/ComfyUI/custom_nodes/ComfyUI-NK2E \
    && git checkout ea321e1603e019f90971c743adce17146756fdac \
    && pip install --no-cache-dir opencv-python-headless

RUN git clone https://github.com/alexw5702-afk/krea2-anypaint.git \
      /opt/ComfyUI/custom_nodes/krea2-anypaint \
    && cd /opt/ComfyUI/custom_nodes/krea2-anypaint \
    && git checkout 675be5a91eadbf8b8997b21c0e8e1848310b9571

RUN git clone https://github.com/ostris/ComfyUI-Krea2-Ostris-Edit.git \
      /opt/ComfyUI/custom_nodes/ComfyUI-Krea2-Ostris-Edit \
    && cd /opt/ComfyUI/custom_nodes/ComfyUI-Krea2-Ostris-Edit \
    && git checkout 7756566160c4a1b24bb1bd9f0ff3ced1a83d7547

RUN git clone https://github.com/capitan01R/ComfyUI-Krea2T-Enhancer.git \
      /opt/ComfyUI/custom_nodes/ComfyUI-Krea2T-Enhancer \
    && cd /opt/ComfyUI/custom_nodes/ComfyUI-Krea2T-Enhancer \
    && git checkout 6666ffb729b2afc8ecdffd3cb6f121555ec948cb

WORKDIR /opt/nvfp4-api
COPY api.py prepare_models.py start.sh ./
RUN chmod +x start.sh

EXPOSE 8691
ENTRYPOINT ["/opt/nvfp4-api/start.sh"]
