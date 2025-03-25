# mrtrix includes fsl, mrtrix, freesurfer, ants, art
FROM mrtrix3/mrtrix3:latest

# Copy uv binaries to install Python
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Add FSL and mrtrix3 to path
RUN echo 'export PATH="/opt/fsl/bin:/opt/mrtrix3/bin:$PATH"' >> /etc/profile.d/path.sh \
&& chmod +x /etc/profile.d/path.sh

# Install Python and pip
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Add a non-root user and set up their home directory
RUN groupadd -r user && useradd --no-log-init -r -g user user \
    && mkdir -p /home/user/.local/lib/python3.10/site-packages \
    && chown -R user:user /home/user

# Set up Python environment for the user
ENV HOME=/home/user
ENV PYTHONPATH=/code/python/cpython-3.10.16-linux-x86_64-gnu/lib/python3.10/site-packages:$HOME/.local/lib/python3.10/site-packages
ENV PATH=$HOME/.local/bin:$PATH

USER user
COPY --chown=user:user --chmod=755 . /code

# Install Python dependencies using UV
RUN mkdir -p /code/.cache/matplotlib
ENV MPLCONFIGDIR=/code/.cache/matplotlib
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy
ENV UV_CACHE_DIR=/code/.cache/uv
ENV UV_PYTHON_INSTALL_DIR=/code/python
WORKDIR /code

# Create virtual environment and install dependencies
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv venv && \
    . .venv/bin/activate && \
    uv pip install --index-url https://pypi.org/simple --extra-index-url https://download.pytorch.org/whl/cpu -e .

# Add virtual environment to PATH
ENV PATH=/code/.venv/bin:$PATH

# Verify torch installation
RUN python3 -c "import torch; print(f'PyTorch version: {torch.__version__}')"

ENTRYPOINT ["/code/scripts/entrypoint.sh"]