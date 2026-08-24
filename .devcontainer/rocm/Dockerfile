ARG ROCM_VERSION=7.2

FROM mambaorg/micromamba:2.1.1 AS micromamba

FROM rocm/dev-ubuntu-24.04:${ROCM_VERSION}-complete

ARG ROCM_VERSION
ARG PY_VERSION=3.12
ARG TORCH_VERSION=2.9.1
ARG AITER_VERSION=0.1.10
ARG AITER_ROCM_VERSION=7.1.1

# Update package lists and install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    clang-format \
    clangd-19 \
    curl \
    emacs \
    git \
    libstdc++-14-dev \
    sudo \
    tmux \
    vim \
    wget \
    zsh \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user
ARG USERNAME=devuser
ARG USER_UID=1003
ARG USER_GID=$USER_UID
ARG MAMBA_USER_ID=${USER_UID}
ARG MAMBA_USER_GID=${USER_GID}
ENV MAMBA_USER=$USERNAME
ENV MAMBA_ROOT_PREFIX="/opt/conda"
ENV MAMBA_EXE="/bin/micromamba"

# Silence warning where the UID and GID are out of the usual range,
# create a non-root user.
RUN sed -i 's/^\(UID_MAX\s*\).*$/\11000000000/' /etc/login.defs && \
    sed -i 's/^\(GID_MAX\s*\).*$/\11000000000/' /etc/login.defs && \
    groupadd --gid $USER_GID $USERNAME && \
    useradd --uid $USER_UID --gid $USER_GID -m $USERNAME && \
    echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME && \
    chmod 0440 /etc/sudoers.d/$USERNAME && \
    usermod -a -G render $USERNAME

RUN echo "set-option -g default-command \"/bin/bash -i\"" >> /home/$USERNAME/.tmux.conf

# Remove default 'ubuntu' user (UID 1000) to prevent devcontainer permission conflicts
RUN rm -rf /home/ubuntu && if grep ubuntu:x:1000:1000 /etc/passwd >/dev/null; then userdel -f -r ubuntu; fi

# Adding micromamba functionality to an existing Docker image requires copying over needed files from the micromamba image
# Refer: https://micromamba-docker.readthedocs.io/en/latest/advanced_usage.html
COPY --from=micromamba "$MAMBA_EXE" "$MAMBA_EXE"
COPY --from=micromamba /usr/local/bin/_activate_current_env.sh /usr/local/bin/_activate_current_env.sh
COPY --from=micromamba /usr/local/bin/_dockerfile_shell.sh /usr/local/bin/_dockerfile_shell.sh
COPY --from=micromamba /usr/local/bin/_entrypoint.sh /usr/local/bin/_entrypoint.sh
COPY --from=micromamba /usr/local/bin/_dockerfile_initialize_user_accounts.sh /usr/local/bin/_dockerfile_initialize_user_accounts.sh
COPY --from=micromamba /usr/local/bin/_dockerfile_setup_root_prefix.sh /usr/local/bin/_dockerfile_setup_root_prefix.sh

RUN /usr/local/bin/_dockerfile_initialize_user_accounts.sh && \
    /usr/local/bin/_dockerfile_setup_root_prefix.sh

# Switch to non-root user
USER $USERNAME
# Set home directory to the user's home directory
WORKDIR /home/$USERNAME

# Set clangd path
ENV PATH="/usr/lib/llvm-19/bin:$PATH"

# Create the new micromamba environment
ARG MAMBA_ENV_NAME=flashinfer-py${PY_VERSION}-torch${TORCH_VERSION}-rocm${ROCM_VERSION}
ENV MAMBA_ENV_NAME=${MAMBA_ENV_NAME}

RUN \
    sed -i 's/^#\(force_color_prompt=yes\)/\1/' /home/$USERNAME/.bashrc && \
    echo '' >> /home/$USERNAME/.bashrc && \
    echo 'eval "$(micromamba shell hook --shell bash)"' >> /home/$USERNAME/.bashrc && \
    echo "micromamba activate ${MAMBA_ENV_NAME}" >> /home/$USERNAME/.bashrc

# Create a new micromamba env and install needed packages for flashinfer development.
# torch uses -f/--find-links (not --index-url) because the radeon repo is a flat
# wheel listing, not a PEP 503 simple index — with --index-url pip requests the
# per-package path (.../torch/), which 404s, and the install fails with "No
# matching distribution found". After installing, assert torch is a ROCm/HIP build
# (torch.version.hip is not None) so the build fails fast if -f ever resolves a
# PyPI CPU/CUDA wheel instead of the radeon ROCm wheel.
RUN /bin/micromamba create -n ${MAMBA_ENV_NAME} python=${PY_VERSION} gtest gmock bash-completion -c conda-forge && \
    /bin/micromamba run -n ${MAMBA_ENV_NAME} pip install --no-cache-dir ninja build "setuptools>=80" "setuptools-scm>=9.2" "packaging>=24" pre-commit numpy pytest pytest-cov pytest-xdist pytest-rerunfailures pybind11 ruff && \
    /bin/micromamba run -n ${MAMBA_ENV_NAME} pip install --no-cache-dir torch==${TORCH_VERSION} -f https://repo.radeon.com/rocm/manylinux/rocm-rel-${ROCM_VERSION}/ && \
    /bin/micromamba run -n ${MAMBA_ENV_NAME} python -c "import torch, sys; hip = torch.version.hip; print(f'torch {torch.__version__} (hip={hip})'); sys.exit('ERROR: installed torch has no ROCm/HIP build (hip is None)' if hip is None else 0)" && \
    /bin/micromamba run -n ${MAMBA_ENV_NAME} pip install amd_aiter==${AITER_VERSION} --extra-index-url https://pypi.amd.com/rocm-${AITER_ROCM_VERSION}/simple
SHELL ["/usr/local/bin/_dockerfile_shell.sh"]

ENTRYPOINT ["/usr/local/bin/_entrypoint.sh"]

CMD ["/bin/bash"]
