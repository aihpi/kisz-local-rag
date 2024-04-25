ARG REGISTRY=quay.io
ARG OWNER=aihpi
ARG JUPYTERHUB_VERSION
ARG BASE_CONTAINER=$REGISTRY/$OWNER/workshop-notebook:pytorch-cuda12-ollama-hub-$JUPYTERHUB_VERSION
FROM $BASE_CONTAINER

# Fix hadolint warnings
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# Install requirements
COPY --chown=${NB_USER}:users requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    rm requirements.txt && \
    # Apply permissions fixes to the Conda directory and the home directory of the notebook user
    fix-permissions "${CONDA_DIR}" && \
    fix-permissions "/home/${NB_USER}"

# Copy the rest of the application into the container
COPY --chown=${NB_USER}:users . /home/${NB_USER}/work

# Switch back to jovyan to avoid accidental container runs as root
USER ${NB_UID}
WORKDIR "${HOME}"
