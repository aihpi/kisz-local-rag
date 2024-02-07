ARG REGISTRY=quay.io
ARG OWNER=ai-hpi
ARG BASE_CONTAINER=$REGISTRY/$OWNER/pytorch-notebook:cuda-latest
FROM $BASE_CONTAINER

# Fix hadolint warnings
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

COPY --from=ollama/ollama /bin/ollama /usr/local/bin/ollama

# Install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    rm requirements.txt && \
    # Apply permissions fixes to the Conda directory and the home directory of the notebook user
    fix-permissions "${CONDA_DIR}" && \
    fix-permissions "/home/${NB_USER}"

# Copy the rest of the application into the container
COPY . /home/${NB_USER}/work

# Fix permissions on /home/${NB_USER} as root
USER root
RUN fix-permissions /home/${NB_USER}/work

# Switch back to jovyan to avoid accidental container runs as root
USER ${NB_UID}
WORKDIR "${HOME}"
