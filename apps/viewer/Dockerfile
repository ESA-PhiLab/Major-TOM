FROM continuumio/miniconda3

# 1. Set up user
RUN useradd -m -u 1000 user

# 2. Set Environment Variables
# We need LD_LIBRARY_PATH for Faiss/MKL.
# We also set SOLARA_PROXY_CACHE_DIR to be safe.
ENV HOME=/home/user \
    PATH=/opt/conda/envs/app_env/bin:$PATH \
    LD_LIBRARY_PATH=/opt/conda/envs/app_env/lib:$LD_LIBRARY_PATH \
    SOLARA_PROXY_CACHE_DIR=/home/user/.solara_cache

# 3. Working Directory
WORKDIR /home/user/app

# 4. Create Conda Env
RUN conda create -n app_env python=3.9 wget -y

# 5. Install Faiss via Conda (Pinned MKL)
# This handles the "libmkl_intel_lp64.so.1" error
RUN conda install -n app_env -c pytorch -c nvidia -c conda-forge faiss-gpu=1.7.4 mkl=2021 blas=1.0=mkl -y && \
    conda clean -ya

# 6. Install Pip Requirements
COPY requirements.txt .
# This runs as root and creates a root-owned /home/user/.cache
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 7. Fix Permissions (THE FIX)
# We recursively chown the ENTIRE home directory.
# This fixes the root-owned .cache folder created by pip in step 6.
RUN chown -R user:user /home/user

# 8. Switch to User
USER user

# 9. Download Files
RUN wget https://huggingface.co/datasets/Major-TOM/Core-S2RGB-SigLIP/resolve/main/extras/faiss/viewer-app/siglip_ivfpq.index -O siglip_ivfpq.index && \
    wget https://huggingface.co/datasets/Major-TOM/Core-S2RGB-SigLIP/resolve/main/extras/faiss/viewer-app/siglip_ivfpq_metadata.parquet -O siglip_ivfpq_metadata.parquet

# 10. Copy App Code
COPY --chown=user *.py *.css ./
COPY --chown=user helpers/* ./helpers/

# 11. Entrypoint
ENTRYPOINT ["solara", "run", "app.py", "--host=0.0.0.0", "--port", "7860", "--production"]