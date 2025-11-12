# ParaK8s

ParaK8s is a framework for organizing and running **large-scale machine learning experiments** on multi-node Kubernetes clusters.  
Its core philosophy is that **all experiments are elements from a very large parameterized function space**: each experiment is fully described by a configuration file specifying data pipelines, model architecture, logging, results tracking, and model registry.

---

## Core Tenets

1. **Experiments as Parameterized Functions**  
   Every experiment is defined by a configuration file that completely describes its behavior and dependencies.

2. **Reproducibility**  
   All experiments are reproducible via configuration files, Docker images, and version-controlled datasets.

3. **Scalability**  
   ParaK8s can orchestrate hundreds of experiments across multi-node Kubernetes clusters, leveraging GPUs where available.

4. **Modularity**  
   Pipelines, models, and data preprocessing functions are designed to be **reusable across experiments**.

5. **Experiment Tracking & Artifact Management**  
   Full integration with **MLflow**, databases (Postgres), and model registries ensures experiments are fully auditable.

6. **Device-Agnostic Preprocessing**  
   Data normalization and preprocessing functions are designed to work consistently across devices and datasets.

---

## Example Experiments

### 1. Pneumonia X-Ray Classifier

**Objective:**  
Classify X-ray images to detect the presence of pneumonia, outputting a probability for each image.

**Key Achievements:**
- Achieved **85% categorical accuracy** and **0.82 AUC score**.
- Utilized **transfer learning** and **data augmentation** using TensorFlow.
- Parallelized preprocessing using **Keras PyDataset**.
- Built **reusable GPU-enabled Docker images**.
- Deployed **100 experiments** across a multi-node Kubernetes cluster.
- Configured **Postgres database**, **MLflow tracking server**, and a local Docker repository.

---

### 2. ECG Abnormality Classifier

**Objective:**  
Detect abnormalities in ECG signals using a multi-class classifier.

**Key Achievements:**
- Built a **DeepCNN model** to classify ECG beats into 5 categories.
- Achieved competitive **AUC, precision, recall, and categorical accuracy**.
- Preprocessed and normalized ECG signals using **TensorFlow operations**, ensuring consistency across devices.
- Managed large datasets with a custom **`ECGDataset`** class (`tf.keras.utils.Sequence`).
- Experiments defined entirely by **configuration files** describing architecture, batch sizes, and data pipelines.
- Deployed multiple experiments on a **multi-node Kubernetes cluster** with GPU support.
- Integrated metrics tracking via **MLflow** and artifact management in a **model registry**.

**Pipeline Overview:**
1. Load and preprocess ECG signals from CSV.
2. Batch and normalize signals using the `ECGDataset` class.
3. Train DeepCNN with **class-weighted loss** for imbalanced datasets.
4. Evaluate with **AUC, precision, recall, and categorical accuracy**.
5. Save trained models and track metrics in **MLflow**.

---

ParaK8s demonstrates **flexibility and reproducibility**, supporting experiments on both **image and time-series data**, while leveraging **modular configurations**, **GPU acceleration**, and **cluster-level orchestration**.
