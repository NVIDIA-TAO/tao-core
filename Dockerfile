# Dockerfile for FTMS dev and prod using multistage build

#####################
# Development stage #
#####################
FROM python:3.12-slim AS dev

# Install system dependencies
RUN apt-get update \
    && apt-get install -y build-essential nginx \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /opt/nvidia/tao-core

# Install dependencies
COPY requirements.txt requirements.txt
RUN pip install --upgrade pip uv && uv pip install -r requirements.txt --system

# Copy nginx config
COPY nvidia_tao_core/microservices/nginx.conf /etc/nginx/

# Set flask development environment variables
ENV FLASK_APP=nvidia_tao_core.microservices.app
ENV FLASK_ENV=development
ENV FLASK_DEBUG=1
ENV PYTHONUNBUFFERED=1

# Expose uWSGI port
EXPOSE 8000

#############################
# Production stage  #
#############################
FROM dev AS prod

# Copy installed python packages from dev stage
COPY --from=dev /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages

# Set working directory again (for clarity)
WORKDIR /opt/nvidia/tao-core

# Copy source code
COPY . .

# Install core wheel and nginx config for production
RUN bash release/python/build_wheel.sh && \
    find dist/ -name "nvidia_tao_core*.whl" -type f | xargs -n 1 pip install && \
    cp nvidia_tao_core/microservices/nginx.conf /etc/nginx/ && \
    chmod +x $(python -c "import nvidia_tao_core; print(nvidia_tao_core.__path__[0])")/microservices/*.py && \
    chmod +x $(python -c "import nvidia_tao_core; print(nvidia_tao_core.__path__[0])")/microservices/*.sh && \
    rm -rf tao-core

RUN python -c "import nvidia_tao_core; print('nvidia_tao_core is installed.')"

# Microservices entrypoint
ENV RUN_CLI=0

CMD ["/bin/bash", "-lc", "if [ \"$RUN_CLI\" = \"1\" ]; then /bin/bash; else /bin/bash $(get-microservice-script); fi"]
