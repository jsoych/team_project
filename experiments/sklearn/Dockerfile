FROM experiment

# Install scikit-learn to experiment environment
RUN conda install -n experiment scikit-learn

# Copy experiment to source directory
COPY src src
WORKDIR /src

# Run experiment
CMD [ "conda", "run", "-n", "experiment", "python", "experiment.py", "with", "config.json"]