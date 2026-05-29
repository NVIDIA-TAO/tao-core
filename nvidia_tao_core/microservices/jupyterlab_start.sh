#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -x
## ADD USER jupyterlab
adduser --disabled-login --disabled-password --gecos "" jupyterlab
usermod -aG sudo jupyterlab
## UPDATE PATH FOR USER jupyterlab
echo 'export PATH="/venv/bin:$HOME/.local/bin:$PATH"' >> /home/jupyterlab/.profile
## ADD KUBE ENVIRONMENT FOR USER jupyterlab
echo "export KUBERNETES_SERVICE_HOST=$KUBERNETES_SERVICE_HOST" >> /home/jupyterlab/.profile
echo "export KUBERNETES_SERVICE_PORT=$KUBERNETES_SERVICE_PORT" >> /home/jupyterlab/.profile
echo "export KUBERNETES_SERVICE_PORT_HTTPS=$KUBERNETES_SERVICE_PORT_HTTPS" >> /home/jupyterlab/.profile
## INSTALL SUDO
apt-get install sudo --yes
echo "jupyterlab ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers
## INSTALL PROCPS
apt-get install procps --yes
## INSTALL APT-UTILS
apt-get install apt-utils --yes
## INSTALL RSYNC
apt-get install rsync --yes
## INSTALL HELM
apt-get install curl --yes
curl https://baltocdn.com/helm/signing.asc | gpg --dearmor | tee /usr/share/keyrings/helm.gpg > /dev/null
apt-get install apt-transport-https --yes
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/helm.gpg] https://baltocdn.com/helm/stable/debian/ all main" | tee /etc/apt/sources.list.d/helm-stable-debian.list
apt-get update
apt-get install helm --yes
## INSTALL AWS CLI
/venv/bin/pip3 install awscli
## FETCH NGC COLLATERALS
cd /opt/api && ngc registry resource download-version $NGC_COLLATERALS
chmod -R 777 /opt/api/$NOTEBOOKS_DIR
## COPY TUTORIALS
su - jupyterlab -c "mkdir -p ~/$NOTEBOOKS_DIR/tao_end2end/"
su - jupyterlab -c "[ -d '/opt/api/$NOTEBOOKS_DIR/tutorials' ] && mv /opt/api/$NOTEBOOKS_DIR/tutorials/* ~/$NOTEBOOKS_DIR/ && rm -rf /opt/api/$NOTEBOOKS_DIR/tutorials"
su - jupyterlab -c "[ -d '/opt/api/$NOTEBOOKS_DIR' ] && mv /opt/api/$NOTEBOOKS_DIR/* ~/$NOTEBOOKS_DIR/tao_end2end/"
## START JUPYTER LAB
apt-get install -y python3-venv
/venv/bin/pip3 install virtualenv
echo 'export PYTHONPATH="/home/jupyterlab/venv/lib/python3.11/site-packages:$PYTHONPATH"' >> /home/jupyterlab/.profile
su - jupyterlab -c "python3 -m virtualenv venv"
su - jupyterlab -c "source /home/jupyterlab/venv/bin/activate && cd ~/$NOTEBOOKS_DIR && /venv/bin/jupyter-lab --ip='0.0.0.0' --no-browser --ServerApp.token='' --ServerApp.password='' --NotebookApp.base_url='/notebook'"
