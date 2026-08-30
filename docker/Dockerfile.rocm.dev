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

# Header donor. The rocm/pytorch base is runtime-only: it carries the pip ROCm
# SDK (_rocm_sdk_core) rather than /opt/rocm, and that SDK ships no math or
# utility headers -- no thrust, rocPRIM, rocBLAS or hipBLASLt. torch 2.12's
# complex.h needs thrust/complex.h, so the JIT cannot compile without them.
FROM rocm/dev-ubuntu-${UBUNTU_VERSION}:${ROCM_VERSION}.0-full AS rocmdev

FROM rocm/pytorch:rocm${ROCM_VERSION}_ubuntu${UBUNTU_VERSION}_py${PY_VERSION}_pytorch_release_${TORCH_VERSION}
ARG ROCM_VERSION
ARG PY_VERSION

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

# Ubuntu 26.04 ships a default `ubuntu` user at UID/GID 1000, which is the most
# common host UID and what the README's $(id -u) yields -- so it has to go
# *before* devuser is created, or groupadd fails with "GID '1000' already
# exists" and the build dies here.
RUN rm -rf /home/ubuntu && \
    if grep -q '^ubuntu:x:1000:1000' /etc/passwd; then userdel -f -r ubuntu; fi

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

# System python3.14 is PEP 668 externally-managed, so everything goes into the
# base image's existing venv -- that is where torch already lives, and creating a
# second one would hide it.
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="$VIRTUAL_ENV/bin:/usr/lib/llvm-19/bin:$PATH"

# Close the three gaps between the pip ROCm SDK and a /opt/rocm install, without
# which every JIT compile dies on `/opt/rocm/bin/amdclang++: not found`.
ENV ROCM_SDK="/opt/venv/lib/python${PY_VERSION}/site-packages/_rocm_sdk_core"
COPY --from=rocmdev /opt/rocm/core-${ROCM_VERSION}/include /opt/rocm-headers/include

#  1. Headers, from the donor stage above.
ENV CPATH="/opt/rocm-headers/include${CPATH:+:$CPATH}"
#  2. Link names: the SDK ships only versioned libs (libamdhip64.so.7), so
#     -lamdhip64 cannot resolve without an unversioned symlink.
#  3. Compiler name: flashinfer invokes $ROCM_HOME/bin/amdclang++, and the SDK
#     ships bin/hipcc plus lib/llvm/bin/clang++ under a different name. The
#     device bitcode also sits at lib/llvm/amdgcn rather than $ROCM_PATH/amdgcn.
RUN set -eu; \
    for so in "$ROCM_SDK"/lib/lib*.so.[0-9]*; do \
        [ -e "$so" ] || continue; \
        base="${so%%.so.*}.so"; [ -e "$base" ] || ln -s "$(basename "$so")" "$base"; \
    done; \
    ln -sf ../lib/llvm/bin/clang++ "$ROCM_SDK/bin/amdclang++"; \
    ln -sf ../lib/llvm/bin/clang "$ROCM_SDK/bin/amdclang"; \
    [ -e "$ROCM_SDK/amdgcn" ] || ln -s lib/llvm/amdgcn "$ROCM_SDK/amdgcn"

# Point at the SDK root, NOT at the grafted header tree: hipcc resolves its own
# clang via $ROCM_PATH/lib/llvm/bin, which only exists under the SDK.
ENV ROCM_PATH="$ROCM_SDK"
ENV ROCM_HOME="$ROCM_SDK"
# Appended rather than assigned so a future base image that sets these for
# torch's own bundled ROCm libs is not clobbered. Today's base sets none of them.
ENV LIBRARY_PATH="${LIBRARY_PATH:+$LIBRARY_PATH:}$ROCM_SDK/lib"
ENV LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+$LD_LIBRARY_PATH:}$ROCM_SDK/lib"

# Hand over the venv *directories* only, then install as the dev user so every
# package added below is owned by them. A recursive chown would instead copy the
# base image's whole 19 GB venv into a new layer.
RUN find "$VIRTUAL_ENV" -type d -exec chown $USER_UID:$USER_GID {} +
USER $USERNAME

# setuptools and setuptools-scm are the in-tree backend's build requirements.
# They must be in the image because the documented install passes
# --no-build-isolation, which is itself required: isolation resolves torch from
# PyPI and pulls a non-ROCm wheel.
RUN pip install --no-cache-dir \
        ninja \
        "setuptools>=80" \
        "setuptools-scm>=9.2" \
        "packaging>=24" \
        pre-commit \
        numpy \
        pytest pytest-cov pytest-xdist pytest-rerunfailures \
        pybind11 \
        ruff \
        filelock && \
    pip install --no-cache-dir "amd-aiter==${AITER_VERSION}" --extra-index-url "${AITER_INDEX}" && \
    python3 -c "import importlib.metadata as m; print('amd-aiter', m.version('amd-aiter'))"

WORKDIR /home/$USERNAME

CMD ["/bin/bash"]
