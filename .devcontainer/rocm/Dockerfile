# ROCm development container for amd-flashinfer.
#
# The base image already provides ROCm, Python and a HIP build of PyTorch inside
# a venv at /opt/venv, so this file adds only developer tooling and AITER.
#
# Two things that are NOT obvious and are load-bearing:
#
#  1. No conda/micromamba layer. Torch lives in the base image's interpreter, and
#     a separate conda environment would shadow it with a torch-less Python.
#     There is also no pip-installable torch for this ROCm release --
#     repo.radeon.com/rocm/manylinux/rocm-rel-7.14/ returns 404 -- so the bundled
#     torch is the only one available.
#  2. Nothing may `import aiter` at build time. It runs rocminfo for arch
#     detection *and* imports triton, which fails with
#     "0 active drivers ([]). There should only be one." when no GPU is attached,
#     and that has no env-var escape. Install it here; verify it at run time.
ARG ROCM_VERSION=7.14
ARG UBUNTU_VERSION=26.04
ARG PY_VERSION=3.14
ARG TORCH_VERSION=2.12.0

FROM rocm/pytorch:rocm${ROCM_VERSION}_ubuntu${UBUNTU_VERSION}_py${PY_VERSION}_pytorch_release_${TORCH_VERSION}

# AITER is pinned to an exact build including the local version segment: this is
# currently the only cp314 wheel published anywhere, and pip will not select a
# local version from a loose specifier.
ARG AITER_VERSION=0.1.16.post3.dev0+g620287969.d20260725
ARG AITER_INDEX=https://rocm.frameworks-nightlies.amd.com/whl-multi-arch/vllm-cdna/

# Fail fast if the base image ever stops shipping a HIP-enabled torch.
RUN python3 -c "import torch, sys; hip = torch.version.hip; print(f'torch {torch.__version__} (hip={hip})'); sys.exit('ERROR: installed torch has no ROCm/HIP build (hip is None)' if hip is None else 0)"

# System dependencies. libstdc++-14-dev and clangd-19 are both still available on
# Ubuntu 26.04, so these names carry over from the 24.04 image unchanged.
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

ARG USERNAME=devuser
ARG USER_UID=1003
ARG USER_GID=$USER_UID

# Silence the warning about out-of-range UID/GID, then create the user.
RUN sed -i 's/^\(UID_MAX\s*\).*$/\11000000000/' /etc/login.defs && \
    sed -i 's/^\(GID_MAX\s*\).*$/\11000000000/' /etc/login.defs && \
    groupadd --gid $USER_GID $USERNAME && \
    useradd --uid $USER_UID --gid $USER_GID -m $USERNAME && \
    echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME && \
    chmod 0440 /etc/sudoers.d/$USERNAME

# The base image has a `video` group but no `render` group, while /dev/dri/render*
# is render-owned on the hosts. Create it so the user can be a member; callers
# must still pass the host's numeric render GID via --group-add, since the
# in-image GID will not generally match the host's.
RUN (getent group render >/dev/null || groupadd -r render) && \
    usermod -a -G render,video $USERNAME

RUN echo "set-option -g default-command \"/bin/bash -i\"" >> /home/$USERNAME/.tmux.conf

# Remove the default 'ubuntu' user (UID 1000) to prevent devcontainer permission
# conflicts. It still exists on Ubuntu 26.04.
RUN rm -rf /home/ubuntu && if grep ubuntu:x:1000:1000 /etc/passwd >/dev/null; then userdel -f -r ubuntu; fi

# System python3.14 is PEP 668 externally-managed, so everything goes into the
# base image's existing venv -- that is where torch already lives, and creating a
# second one would hide it.
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="$VIRTUAL_ENV/bin:/usr/lib/llvm-19/bin:$PATH"

RUN pip install --no-cache-dir \
        ninja \
        "setuptools>=80" \
        "setuptools-scm>=9.2" \
        pre-commit \
        numpy \
        pytest pytest-cov pytest-xdist pytest-rerunfailures \
        pybind11 \
        ruff \
        filelock && \
    pip install --no-cache-dir "amd_aiter==${AITER_VERSION}" --extra-index-url "${AITER_INDEX}" && \
    python3 -c "import importlib.metadata as m; print('amd-aiter', m.version('amd-aiter'))"

# Editable installs write into the venv, so hand it to the dev user.
RUN chown -R $USER_UID:$USER_GID $VIRTUAL_ENV

USER $USERNAME
WORKDIR /home/$USERNAME

CMD ["/bin/bash"]
