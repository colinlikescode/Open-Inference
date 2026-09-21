#!/bin/sh
# Runs only on user-supplied GPU machines. Installs local runtime prerequisites, never VMs.
set -eu
test "$(uname -s)" = Linux || { echo 'GPU workers must run Linux' >&2; exit 2; }
if [ "$(id -u)" = 0 ]; then
    elevate() { "$@"; }
else
    elevate() { sudo -n "$@"; }
fi

if ! command -v docker >/dev/null 2>&1 || ! command -v python3 >/dev/null 2>&1; then
    if command -v apt-get >/dev/null 2>&1; then
        elevate apt-get update
        elevate env DEBIAN_FRONTEND=noninteractive apt-get install -y docker.io python3 curl ca-certificates gnupg
    elif command -v dnf >/dev/null 2>&1; then
        elevate dnf install -y docker python3 curl ca-certificates gnupg2
    else
        echo 'Automatic bootstrap supports apt-get and dnf. Install Docker and Python 3 on this worker.' >&2
        exit 2
    fi
    elevate systemctl enable --now docker
fi

if ! command -v nvidia-ctk >/dev/null 2>&1; then
    if command -v apt-get >/dev/null 2>&1; then
        elevate apt-get update
        elevate env DEBIAN_FRONTEND=noninteractive apt-get install -y curl ca-certificates gnupg
        setup_dir=$(mktemp -d)
        trap 'rm -rf "$setup_dir"' EXIT
        curl --fail --silent --show-error --location https://nvidia.github.io/libnvidia-container/gpgkey -o "$setup_dir/key"
        gpg --batch --dearmor --output "$setup_dir/keyring.gpg" "$setup_dir/key"
        elevate install -m 0644 "$setup_dir/keyring.gpg" /usr/share/keyrings/opensandbox-nvidia.gpg
        curl --fail --silent --show-error --location https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list -o "$setup_dir/repository"
        sed 's#deb https://#deb [signed-by=/usr/share/keyrings/opensandbox-nvidia.gpg] https://#' "$setup_dir/repository" > "$setup_dir/signed-repository"
        elevate install -m 0644 "$setup_dir/signed-repository" /etc/apt/sources.list.d/opensandbox-nvidia.list
        elevate apt-get update
        elevate env DEBIAN_FRONTEND=noninteractive apt-get install -y nvidia-container-toolkit
    elif command -v dnf >/dev/null 2>&1; then
        elevate dnf install -y curl
        elevate curl --fail --silent --show-error --location https://nvidia.github.io/libnvidia-container/stable/rpm/nvidia-container-toolkit.repo -o /etc/yum.repos.d/opensandbox-nvidia.repo
        elevate dnf install -y nvidia-container-toolkit
    else
        echo 'Install NVIDIA Container Toolkit on this worker before using --no-bootstrap.' >&2
        exit 2
    fi
fi

if ! docker --host unix:///var/run/docker.sock info >/dev/null 2>&1 && ! elevate docker --host unix:///var/run/docker.sock info >/dev/null 2>&1; then
    elevate systemctl start docker
fi
if docker --host unix:///var/run/docker.sock info >/dev/null 2>&1; then
    docker_control() { docker --host unix:///var/run/docker.sock "$@"; }
else
    docker_control() { elevate docker --host unix:///var/run/docker.sock "$@"; }
fi
if ! docker_control info --format '{{json .Runtimes}}' | python3 -c 'import json,sys; sys.exit(0 if "nvidia" in json.load(sys.stdin) else 1)'; then
    if [ -n "$(docker_control ps -q)" ]; then
        echo 'NVIDIA Docker runtime needs configuration, but other containers are running. Configure it during a maintenance window, then retry.' >&2
        exit 2
    fi
    elevate nvidia-ctk runtime configure --runtime=docker
    elevate systemctl restart docker
fi
docker_control info --format '{{json .ServerVersion}}'
