<div align=center><img width="60%" height="60%" src="assets/logo.png"/></div>

--------------------------------------------------------------------------------
![Build Status](https://travis-ci.com/DarkGeekMS/Retratista.svg?token=xzMWZ6kxunjrsLmkSrpx&branch=main)
[![GPLv3 license](https://img.shields.io/badge/License-GPLv3-blue.svg)](http://perso.crans.org/besson/LICENSE.html)

Retratista is a web-based application that enables the user to generate and manipulate a realistic human face from bare description.

## System Workflow

<div align=center><img width="80%" height="80%" src="assets/system-design.png"/></div>

<div align="center">
Figure(1): Complete block diagram showing the whole system design and flow between different modules.
</div><br>

## Application Design

<div align=center><img width="50%" height="50%" src="assets/app-design.png"/></div>

<div align="center">
Figure(2): Web application design.
</div><br>

## Usage

### Docker

-   Make sure you installed :
    -   Nvidia CUDA 11.1 + CUDNN8
    -   Docker CE.
    -   Docker Compose.
    -   [Nvidia Docker V2](https://github.com/NVIDIA/nvidia-docker)

-   Start application server using __docker-compose__ :
    ```bash
    bash scripts/docker_run.sh
    ```

### Native

-   Make sure you installed :
    -   Nvidia CUDA 11.1 + CUDNN8
    -   python3 + python3-pip (Anaconda can be used as well).

-   Install python dependencies :
    ```bash
    pip3 install -r requirements.txt
    ```

-   Download model weights :
    ```bash
    bash scripts/download_weights.sh
    ```

-   Run application server :
    ```bash
    python3 run.py production 5000
    ```

-   Run application server (in debug mode):
    ```bash
    python3 run.py development 5000
    ```
