.. meta::
  :description: installing FlashInfer for ROCm
  :keywords: installation instructions, Docker, AMD, ROCm, FlashInfer

.. _flashinfer-on-rocm-installation:

********************************************************************
FlashInfer on ROCm installation
********************************************************************

System requirements
====================================================================

To use FlashInfer `0.2.5 <https://github.com/flashinfer-ai/flashinfer/releases/tag/v0.2.5>`__, you need the following prerequisites:

- **ROCm version:** `7.1.1 <https://rocm.docs.amd.com/en/docs-7.1.1/>`__
- **Operating system:** Ubuntu 24.04
- **GPU platform:** AMD Instinct™ MI325X, MI300X
- **PyTorch:** `2.8.0 <https://github.com/ROCm/pytorch/releases/tag/v2.8.0>`__
- **Python:** `3.12 <https://www.python.org/downloads/release/python-3129/>`__

Install FlashInfer
================================================================================

To install FlashInfer on ROCm, you have the following options:

* :ref:`using-docker-with-flashinfer-pre-installed` **(recommended)**
* :ref:`flashinfer-pip-install`
* :ref:`build-flashinfer-rocm-docker-image`

.. _using-docker-with-flashinfer-pre-installed:

Use a prebuilt Docker image with FlashInfer pre-installed
--------------------------------------------------------------------------------------

Docker is the recommended method to set up a FlashInfer environment, as it avoids 
potential installation issues.  The tested, prebuilt image includes FlashInfer, PyTorch, 
ROCm, and other dependencies.

1. Pull the Docker image.

   .. code-block:: bash

      docker pull rocm/flashinfer:flashinfer-0.2.5.amd2_rocm7.1.1_ubuntu24.04_py3.12_pytorch2.8

2. Start a Docker container using the image.

   .. code-block:: bash

      docker run -it --rm \
      --privileged \
      --network=host --device=/dev/kfd \
      --device=/dev/dri --group-add video \
      --name=my_flashinfer --cap-add=SYS_PTRACE \
      --security-opt seccomp=unconfined \
      --ipc=host --shm-size 16G \
      rocm/flashinfer:flashinfer-0.2.5.amd2_rocm7.1.1_ubuntu24.04_py3.12_pytorch2.8

3. The above step will create a Docker container with FlashInfer pre-installed. During this process, the Dockerfile will have a pre-installed and configured micromamba environment with FlashInfer available inside. To use FlashInfer, activate the micromamba environment.

   .. code-block:: bash

      micromamba activate base

.. _flashinfer-pip-install:

Install FlashInfer using pip
--------------------------------------------------------------------------------------

Use a base PyTorch Docker image and follow these steps to install FlashInfer using pip.  

1. Pull the base ROCm PyTorch Docker image.

   .. code-block:: bash

      docker pull rocm/pytorch:rocm7.1.1_ubuntu24.04_py3.12_pytorch_release_2.8.0

2. Start a Docker container using the image.

   .. code-block:: bash

      docker run -it --rm \
      --privileged \
      --network=host --device=/dev/kfd \
      --device=/dev/dri --group-add video \
      --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
      --ipc=host --shm-size 128G \
      rocm/pytorch:rocm7.1.1_ubuntu24.04_py3.12_pytorch_release_2.8.0

3. After setting up the container, install FlashInfer from the AMD-hosted `PyPI repository <https://pypi.amd.com/rocm-7.1.1/simple/amd-flashinfer/>`__.

   .. code-block:: bash

      pip install amd-flashinfer --index-url https://pypi.amd.com/rocm-7.1.1/simple

.. _build-flashinfer-rocm-docker-image:

Build from source
--------------------------------------------------------------------------------------

FlashInfer on ROCm can be run directly by setting up a Docker container from scratch. 
A Dockerfile is provided in the `https://github.com/AMD-Ecosystem/flashinfer/blob/amd-integration/.devcontainer/rocm/Dockerfile <https://github.com/AMD-Ecosystem/flashinfer/blob/amd-integration/.devcontainer/rocm/Dockerfile>`__ repository to help you get started.

1. Clone the `https://github.com/AMD-Ecosystem/flashinfer <https://github.com/AMD-Ecosystem/flashinfer>`__ repository.

   .. code-block:: bash
      
      git clone https://github.com/AMD-Ecosystem/flashinfer.git

2. Enter the directory and build the Dockerfile to create a Docker image.
   
   .. code-block:: bash
      
      cd flashinfer
      docker build \
      --build-arg USERNAME=$USER \
      --build-arg USER_UID=$(id -u) \
      --build-arg USER_GID=$(id -g) \
      -f .devcontainer/rocm/Dockerfile \
      -t rocm-flashinfer-dev .

3. Start a Docker container using the image.

   .. code-block:: bash
      
      docker run -it --rm \
      --privileged --network=host --device=/dev/kfd \
      --device=/dev/dri --group-add video \
      --cap-add=SYS_PTRACE \
      --security-opt seccomp=unconfined \
      --ipc=host --shm-size 16G \
      -v $PWD:/workspace \
      rocm-flashinfer-dev

4. Once you are inside the container, the micromamba environment is automatically activated. You can now install FlashInfer inside it.

   .. code-block:: bash

      cd /workspace
      FLASHINFER_HIP_ARCHITECTURES=gfx942 python -m pip wheel . --wheel-dir=./dist/ --no-deps --no-build-isolation -v
      cd dist && pip install amd_flashinfer-*.whl

Test the FlashInfer installation
======================================================================================

Once you have installed FlashInfer, run the following command. If it outputs ``0.2.5+amd.2``, then FlashInfer is installed correctly. You can now use FlashInfer in your projects.

.. code-block:: bash

   python -c "import flashinfer; print(flashinfer.__version__)"
