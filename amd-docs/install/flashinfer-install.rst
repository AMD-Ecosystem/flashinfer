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

- **ROCm version:** `6.4.1 <https://rocm.docs.amd.com/en/docs-6.4.1/>`__
- **Operating system:** Ubuntu 24.04
- **GPU platform:** AMD Instinct™ MI300X
- **PyTorch:** `2.7.1 <https://github.com/ROCm/pytorch/releases/tag/v2.7.1>`__
- **Python:** `3.12 <https://www.python.org/downloads/release/python-3129/>`__

Install FlashInfer
================================================================================

To install FlashInfer on ROCm, you have the following options:

* :ref:`using-docker-with-flashinfer-pre-installed` **(recommended)**
* :ref:`build-flashinfer-rocm-docker-image`
* :ref:`flashinfer-pip-install`

.. _using-docker-with-flashinfer-pre-installed:

Use a prebuilt Docker image with FlashInfer pre-installed
--------------------------------------------------------------------------------------

Docker is the recommended method to set up a FlashInfer environment, as it avoids 
potential installation issues.  The tested, prebuilt image includes FlashInfer, PyTorch, 
ROCm, and other dependencies.

1. Pull the Docker image:

   .. code-block:: bash

      docker pull rocm/flashinfer:flashinfer-0.2.5_rocm6.4_ubuntu24.04_py3.12_pytorch2.7


2. Launch and connect to the container:

   .. code-block:: bash

      docker run -it --rm \
      --privileged -v ./:/app \
      --network=host --device=/dev/kfd \
      --device=/dev/dri --group-add video \
      --name=my_flashinfer --cap-add=SYS_PTRACE \
      --security-opt seccomp=unconfined \
      --ipc=host --shm-size 16G \
      rocm/flashinfer:flashinfer-0.2.5_rocm6.4_ubuntu24.04_py3.12_pytorch2.7

.. _build-flashinfer-rocm-docker-image:

Build your own Docker image
--------------------------------------------------------------------------------------

FlashInfer on ROCm can be run directly by setting up a Docker container from scratch. 
A Dockerfile is provided in the `https://github.com/AMD-Ecosystem/flashinfer/blob/amd-integration/.devcontainer/rocm/Dockerfile <https://github.com/AMD-Ecosystem/flashinfer/blob/amd-integration/.devcontainer/rocm/Dockerfile>`__ repository to help you get started.

1. Clone the `https://github.com/AMD-Ecosystem/flashinfer <https://github.com/AMD-Ecosystem/flashinfer>`__ repository:

   .. code-block:: bash
      
      git clone https://github.com/AMD-Ecosystem/flashinfer.git

2. Enter the directory and build the Dockerfile:
   
   .. code-block:: bash
      
      cd flashinfer
      docker build -t rocm/flashinfer:flashinfer-0.2.5_rocm6.4_ubuntu24.04_py3.12_pytorch2.7

3. Run the Docker container:

   .. code-block:: bash
      
      docker run -it --device=/dev/kfd --device=/dev/dri --group-add video rocm/flashinfer:flashinfer-0.2.5_rocm6.4_ubuntu24.04_py3.12_pytorch2.7

4. The above step will create a Docker container with FlashInfer pre-installed. During this process, the Dockerfile will have pre-installed and configured a micromamba environment named ``flashinfer-py3.12-torch2.7.1-rocm6.4.1``.

.. _flashinfer-pip-install:

Install FlashInfer using pip
--------------------------------------------------------------------------------------

Use a base PyTorch Docker image and follow these steps to install FlashInfer using pip.  

1. Pull the base ROCm PyTorch Docker image:

   .. code-block:: bash

      docker pull rocm/pytorch:rocm6.4.1_ubuntu24.04_py3.12_pytorch_release_2.7.1
   
2. Change the ``<container name>`` and then use the following command:

   .. code-block:: bash
      
      docker run -it --privileged --network=host --device=/dev/kfd --device=/dev/dri --group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined --ipc=host --shm-size 128G --name=<container name> rocm/pytorch:rocm6.4.1_ubuntu24.04_py3.12_pytorch_release_2.7.1

3. After setting up the container, install FlashInfer from the AMD-hosted `PyPI repository <https://pypi.amd.com/simple/>`__.

   .. code-block:: bash

      pip install flashinfer==0.2.5.post10 --index-url=https://pypi.amd.com/simple

Test the FlashInfer installation
======================================================================================

Once you have the Docker container running, start using FlashInfer by following these steps:

1. Activate the micromamba environment:

   .. note::

      If you followed :ref:`flashinfer-pip-install`, you do not need to activate the micromamba environment.
      If you followed :ref:`using-docker-with-flashinfer-pre-installed` or :ref:`build-flashinfer-rocm-docker-image`, don't forget this step.

   .. code-block:: bash
      
      micromamba activate flashinfer-py3.12-torch2.7.1-rocm6.4.1

2. Enter the FlashInfer directory:

   .. note::

      If you followed :ref:`flashinfer-pip-install`, ensure you git clone the repository first.

      .. code-block:: bash
         
         git clone https://github.com/AMD-Ecosystem/flashinfer.git 

   .. code-block:: bash
      
      cd flashinfer/
   
3. Run the example provided in the ``flashinfer/examples`` directory. This example runs ``Batch Decode`` and then verifies the output. 

   .. code-block:: bash

      python examples/test_batch_decode_example.py

3. If FlashInfer was installed correctly, you should see the following output:
   
   .. code-block:: bash

      PASS

4. The above output indicates that FlashInfer is installed correctly. You can now use FlashInfer in your projects.
