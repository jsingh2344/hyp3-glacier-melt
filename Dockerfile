FROM condaforge/mambaforge:latest

LABEL org.opencontainers.image.title="HyP3 glacier-melt "
LABEL org.opencontainers.image.description="Melt extents and snowlines from RTC products"
LABEL org.opencontainers.image.vendor="Alaska Satellite Facility"
LABEL org.opencontainers.image.authors="jsingh2344 <jadens@andrew.cmu.edu>"
LABEL org.opencontainers.image.licenses="BSD-3-Clause"
LABEL org.opencontainers.image.url="https://github.com/jsingh2344/hyp3-repo"
LABEL org.opencontainers.image.source="https://github.com/jsingh2344/hyp3-repo"
LABEL org.opencontainers.image.documentation="https://hyp3-docs.asf.alaska.edu"

ARG DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=true

RUN apt-get update && apt-get install -y --no-install-recommends unzip vim && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

ARG CONDA_UID=1000
ARG CONDA_GID=1000

RUN groupadd -g "${CONDA_GID}" --system conda && \
    useradd -l -u "${CONDA_UID}" -g "${CONDA_GID}" --system -d /home/conda -m -s /bin/bash conda && \
    chown -R conda:conda /opt && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> /home/conda/.profile && \
    echo "conda activate base" >> /home/conda/.profile

SHELL ["/bin/bash", "-l", "-c"]
WORKDIR /hyp3-glacier-melt

COPY . /hyp3-glacier-melt/

RUN mamba env create -f /hyp3-glacier-melt/environment.yml && \
    conda clean -afy && \
    sed -i 's/conda activate base/conda activate hyp3-glacier-melt/g' /home/conda/.profile && \
    rm -rf /hyp3-glacier-melt/build /hyp3-glacier-melt/dist /hyp3-glacier-melt/*.egg-info && \
    conda run -n hyp3-glacier-melt python -m pip install --no-cache-dir /hyp3-glacier-melt && \
    chown -R ${CONDA_UID}:${CONDA_GID} /hyp3-glacier-melt /home/conda /opt/conda

USER ${CONDA_UID}

ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "hyp3-glacier-melt", "python", "-m", "hyp3_glacier_melt"]
CMD ["-h"]
