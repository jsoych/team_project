FROM experiment

# Install scikit-learn to experiment environment
RUN conda install -n experiment scikit-learn

# Copy experiment to source directory
COPY src src
WORKDIR /src

# Add permissons to run script
RUN chmod +x run.sh

# Run experiment
CMD [ "conda", "run", "-n", "experiment", "python", "experiment.py" ]