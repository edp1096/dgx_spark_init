FROM dgx-flux2-klein-nvfp4:4b

WORKDIR /opt/zimage-api
COPY api.py prepare_models.py start.sh ./
RUN chmod +x start.sh

EXPOSE 8703
ENTRYPOINT ["/opt/zimage-api/start.sh"]
