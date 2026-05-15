# syntax=docker/dockerfile:1.7
# 统一开发/训练镜像: 一个容器同时装下两个项目的 venv
#
#   /app/openpi    pi0.5 推理        uv venv @ /opt/venvs/openpi   Python 3.11
#   /app/RoboTwin  RoboTwin + SAC    conda env RoboTwin            Python 3.10
#
# 两个 venv 互不干扰. compose 起两个 service, 共用这个 image, 各自用自己的 venv.
#
# 构建时要求 RoboTwin 和 openpi side-by-side:
#   parent/
#     ├── openpi/       <- your drift branch
#     └── RoboTwin/     <- 启动 docker compose 的入口
#
# 构建 (从 parent 目录):
#   cd parent
#   docker build . -f RoboTwin/docker/speedtune.Dockerfile -t speedtune:latest
# 或通过 docker compose (见 docker/compose.yml):
#   cd RoboTwin && docker compose -f docker/compose.yml build

FROM nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV CONDA_DIR=/opt/conda
ENV PATH=$CONDA_DIR/bin:$PATH
ENV NVIDIA_DRIVER_CAPABILITIES=compute,graphics,utility
ENV PYOPENGL_PLATFORM=egl

# ------------------------------------------------------------------
# Mirror knobs (build args). 默认走清华源, 国外环境传:
#   --build-arg MINIFORGE_URL=https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
#   --build-arg PIP_INDEX_URL=https://pypi.org/simple
#   --build-arg PYTORCH3D_GIT=https://github.com/facebookresearch/pytorch3d.git
#   --build-arg CUROBO_GIT=https://github.com/NVlabs/curobo.git
# ------------------------------------------------------------------
ARG MINIFORGE_URL=https://mirrors.tuna.tsinghua.edu.cn/github-release/conda-forge/miniforge/LatestRelease/Miniforge3-Linux-x86_64.sh
ARG PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
ARG PIP_TRUSTED_HOST=pypi.tuna.tsinghua.edu.cn
ARG PYTORCH3D_GIT=https://github.com/facebookresearch/pytorch3d.git
ARG CUROBO_GIT=https://github.com/NVlabs/curobo.git
ENV PIP_INDEX_URL=$PIP_INDEX_URL
ENV PIP_TRUSTED_HOST=$PIP_TRUSTED_HOST
ENV UV_INDEX_URL=$PIP_INDEX_URL

# ------------------------------------------------------------------
# 1. system deps (GL / ffmpeg / git / build tools)
# ------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        git git-lfs ca-certificates curl wget bzip2 build-essential \
        cmake ninja-build clang \
        libegl1 libgl1 libgles2 libglvnd0 libglx0 libopengl0 \
        libglfw3 libglfw3-dev \
        libx11-6 libxext6 libxrandr2 libxinerama1 libxcursor1 libxi6 \
        ffmpeg linux-headers-generic \
        && rm -rf /var/lib/apt/lists/*

# Make git fetches retry on flaky links instead of dying mid-stream.
RUN git config --system http.lowSpeedLimit 1000 && \
    git config --system http.lowSpeedTime 60 && \
    git config --system http.postBuffer 1048576000

# ------------------------------------------------------------------
# 2. miniforge (conda) + uv
# ------------------------------------------------------------------
RUN curl -fsSL --retry 8 --retry-delay 5 --retry-all-errors \
        "$MINIFORGE_URL" -o /tmp/miniforge.sh && \
    bash /tmp/miniforge.sh -b -p $CONDA_DIR && \
    rm /tmp/miniforge.sh && \
    conda config --set always_yes yes --set changeps1 no

# 让 conda 的 conda-forge / defaults 走清华源 (PIP_INDEX_URL 用国内时同步切)
RUN if echo "$PIP_INDEX_URL" | grep -q "tuna.tsinghua"; then \
        conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main && \
        conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge && \
        conda config --set show_channel_urls yes ; \
    fi

COPY --from=ghcr.io/astral-sh/uv:0.5.1 /uv /uvx /usr/local/bin/

# ------------------------------------------------------------------
# 3. openpi venv (Python 3.11, uv 管理)
#
# build context 是 parent 目录, 所以 openpi/ 的文件通过 "openpi/xxx" 路径访问
# ------------------------------------------------------------------
ENV UV_LINK_MODE=copy
ENV OPENPI_VENV=/opt/venvs/openpi
ENV UV_PROJECT_ENVIRONMENT=$OPENPI_VENV
RUN uv venv --python 3.11.9 $OPENPI_VENV

WORKDIR /tmp/openpi_build
# 只 COPY 锁依赖需要的东西; 源码走 compose 挂载
COPY openpi/pyproject.toml ./pyproject.toml
COPY openpi/uv.lock ./uv.lock
COPY openpi/packages/openpi-client/pyproject.toml ./packages/openpi-client/pyproject.toml
COPY openpi/packages/openpi-client/src ./packages/openpi-client/src

RUN GIT_LFS_SKIP_SMUDGE=1 \
    bash -c 'for i in 1 2 3 4 5; do \
        uv sync --frozen --no-install-project --no-dev && exit 0; \
        echo "[uv sync] attempt $i failed, retrying in 10s..."; sleep 10; \
    done; exit 1'

# transformers_replace: openpi 对 transformers 源码的 in-place patch
COPY openpi/src/openpi/models_pytorch/transformers_replace /tmp/openpi_tr_replace
RUN $OPENPI_VENV/bin/python -c \
        "import transformers, os; print(os.path.dirname(transformers.__file__))" \
        | xargs -I{} cp -r /tmp/openpi_tr_replace/. {}/ && \
    rm -rf /tmp/openpi_tr_replace /tmp/openpi_build

# Extras for tools/convert_robotwin_to_lerobot.py (reads RoboTwin hdf5)
RUN $OPENPI_VENV/bin/pip install --no-cache-dir h5py tqdm

# ------------------------------------------------------------------
# 4. RoboTwin conda env (Python 3.10)
# ------------------------------------------------------------------
RUN conda create -n RoboTwin python=3.10 -y && conda clean -afy
ENV ROBOTWIN_VENV=$CONDA_DIR/envs/RoboTwin

COPY RoboTwin/script/requirements.txt /tmp/robotwin_requirements.txt
RUN $ROBOTWIN_VENV/bin/pip install --no-cache-dir -r /tmp/robotwin_requirements.txt && \
    rm /tmp/robotwin_requirements.txt

# pytorch3d (stable, no build isolation)
RUN $ROBOTWIN_VENV/bin/pip install --no-cache-dir \
        "git+${PYTORCH3D_GIT}@stable" --no-build-isolation

# RL 通信依赖
RUN $ROBOTWIN_VENV/bin/pip install --no-cache-dir \
        websockets msgpack msgpack-numpy tensorboard

# sapien / mplib 源码 patch (对齐 RoboTwin/script/_install.sh 的 sed)
RUN URDF="$($ROBOTWIN_VENV/bin/python -c 'import sapien, os; print(os.path.join(os.path.dirname(sapien.__file__), "wrapper", "urdf_loader.py"))')" && \
    sed -i -E 's/("r")(\))( as)/\1, encoding="utf-8") as/g' "$URDF"
RUN PLAN="$($ROBOTWIN_VENV/bin/python -c 'import mplib, os; print(os.path.join(os.path.dirname(mplib.__file__), "planner.py"))')" && \
    sed -i -E 's/(if np.linalg.norm\(delta_twist\) < 1e-4 )(or collide )(or not within_joint_limit:)/\1\3/g' "$PLAN"

# curobo 装在 /opt 下, 不进入 /app (防止被挂载覆盖). 运行时通过 symlink 暴露到 envs/curobo
RUN git clone --branch v0.7.8 --depth 1 "$CUROBO_GIT" /opt/curobo && \
    cd /opt/curobo && \
    $ROBOTWIN_VENV/bin/pip install --no-cache-dir -e . --no-build-isolation

# ------------------------------------------------------------------
# 5. workdir + bootstrap entrypoint
# ------------------------------------------------------------------
WORKDIR /app

# 两份源码的预期挂载点:  /app/openpi  /app/RoboTwin
# entrypoint: 每次启动时若 RoboTwin 里没有 envs/curobo (被挂载覆盖), 建一个 symlink
RUN mkdir -p /opt/bootstrap
COPY RoboTwin/docker/entrypoint.sh /opt/bootstrap/entrypoint.sh
RUN chmod +x /opt/bootstrap/entrypoint.sh

ENTRYPOINT ["/opt/bootstrap/entrypoint.sh"]
CMD ["/bin/bash"]
