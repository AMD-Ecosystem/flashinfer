.. meta::
  :description: installing FlashInfer for ROCm
  :keywords: installation instructions, Docker, AMD, ROCm, FlashInfer

.. _flashinfer-on-rocm-installation:

********************************************************************
FlashInfer on ROCm installation
********************************************************************

System requirements
====================================================================

To use FlashInfer `0.6.18 <https://github.com/flashinfer-ai/flashinfer/releases/tag/v0.6.18>`__ on ROCm, you need the following prerequisites:

- **ROCm version:** `10.1.0 <https://rocm.docs.amd.com/en/docs-10.1.0/>`__
- **Operating system:** Ubuntu 24.04
- **GPU platform:** AMD Instinct™ MI300X, MI325X, MI355X
- **PyTorch:** `2.12.0 <https://github.com/ROCm/pytorch/tree/release/2.12>`__
- **Python:** `3.12 <https://www.python.org/downloads/release/python-31215/>`__
- **AITER:** `0.1.21.post2 <https://github.com/ROCm/aiter/releases/tag/v0.1.21.post2>`__ (source build)

Install FlashInfer
====================================================================

FlashInfer on ROCm can be run directly by setting up a Docker container from scratch.

Build FlashInfer from source
--------------------------------------------------------------------

1. Clone the `https://github.com/AMD-Ecosystem/flashinfer <https://github.com/AMD-Ecosystem/flashinfer>`__ repository.

   .. code-block:: bash

      git clone https://github.com/AMD-Ecosystem/flashinfer.git

2. Enter the directory and build the Dockerfile (`https://github.com/AMD-Ecosystem/flashinfer/blob/amd-integration/docker/Dockerfile.rocm <https://github.com/AMD-Ecosystem/flashinfer/blob/amd-integration/docker/Dockerfile.rocm>`__) to create a Docker image.

   .. code-block:: bash

      cd flashinfer
      docker build \
        -t rocm-flashinfer-dev \
        -f docker/Dockerfile.rocm . \
        --build-arg USERNAME=$USER \
        --build-arg USER_UID=$(id -u) \
        --build-arg USER_GID=$(id -g)

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

4. Once you are inside the container, you can now install FlashInfer inside it.

   .. code-block:: bash

      cd /workspace
      python -m pip install --no-build-isolation -ve .

Test the FlashInfer installation
======================================================================================

Verify that FlashInfer is installed correctly.

.. code-block:: bash

   python -c "import flashinfer; print(flashinfer.__version__)"

Expected output:

.. code-block:: text

   0.6.18+amd.2

If you see the version string above, FlashInfer ``0.6.18`` has been installed successfully. You can now use FlashInfer in your projects.

AITER Support
======================================================================================

FlashInfer on ROCm has experimental support to use `https://github.com/ROCm/aiter <https://github.com/ROCm/aiter>`__ as a backend. AITER is a library for efficient attention operations.

The development image built above already includes the validated AITER wheel; no separate AITER installation is needed. Install AITER separately only when assembling your own environment.
Use one of the following options to install AITER:

* :ref:`aiter-build-from-source`
* :ref:`aiter-install-wheel-pypi-amd`

.. _aiter-build-from-source:

Install AITER by building from source
--------------------------------------------------------------------------------------

Use the following command to install AITER by building from source.

.. code-block:: bash

   git clone --recursive https://github.com/ROCm/aiter.git
   cd aiter
   python3 setup.py develop

.. _aiter-install-wheel-pypi-amd:

Install the AITER wheels package using pip
--------------------------------------------------------------------------------------

The wheels package is hosted on the AMD PyPI repository. Use the following command to install AITER with ``pip``.

.. code-block:: bash

   pip install amd-aiter --index-url https://pypi.amd.com/simple/
