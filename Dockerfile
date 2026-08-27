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

RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl unzip vim && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# RGI 7.0 Alaska glacier outlines (CC BY 4.0).
# Dataset DOI: https://doi.org/10.5067/F6JMOVY5NAVZ
ARG RGI_ALASKA_URL="https://ihp-wins.unesco.org/dataset/33a5017a-e6e9-43cc-82d6-62da7fbb74d8/resource/87ed4714-cdcf-45bd-9d1e-827082861656/download/rgi2000-v7.0-g-01_alaska.zip"
ARG RGI_ALASKA_SHA256="23af9038cf28845a476bf3bf9ddd9a2dfb417921ff167d50a6a9cc0e16d4ee55"
ENV RGI_ROOT=/opt/rgi
ENV RGI_SHAPEFILE=/opt/rgi/RGI2000-v7.0-G-01_alaska/RGI2000-v7.0-G-01_alaska.shp

RUN mkdir -p /opt/rgi/RGI2000-v7.0-G-01_alaska && \
    curl -L --fail --retry 3 "${RGI_ALASKA_URL}" -o /tmp/rgi-alaska.zip && \
    echo "${RGI_ALASKA_SHA256}  /tmp/rgi-alaska.zip" | sha256sum --check - && \
    unzip -q /tmp/rgi-alaska.zip -d /opt/rgi/RGI2000-v7.0-G-01_alaska && \
    test -f "${RGI_SHAPEFILE}" && \
    rm /tmp/rgi-alaska.zip

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
    chown -R ${CONDA_UID}:${CONDA_GID} /hyp3-glacier-melt /home/conda /opt/conda /opt/rgi

USER ${CONDA_UID}

ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "hyp3-glacier-melt", "python", "-m", "hyp3_glacier_melt"]
CMD ["-h"]
