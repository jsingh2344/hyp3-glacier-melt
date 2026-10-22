FROM ghcr.io/jsingh2344/hyp3-glacier-melt:test
COPY src /hyp3-glacier-melt/src
ENV PYTHONPATH=/hyp3-glacier-melt/src
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "hyp3-glacier-melt", "python", "-m", "hyp3_glacier_melt.doy_workflow"]
