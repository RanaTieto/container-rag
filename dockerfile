FROM ollama/ollama:latest

# Set up an entrypoint script to pull the model at runtime
COPY entrypoint.sh /root/entrypoint.sh
RUN chmod +x /root/entrypoint.sh

RUN bash /root/entrypoint.sh

