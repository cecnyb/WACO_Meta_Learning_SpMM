# WACO With Meta Learning
Workload-Aware Co-Optimization for a sparse tensor program with meta learning.

This repository includes an artifact for ["WACO: Learning workload-aware co-optimization of the format and schedule of a sparse tensor program"](https://dl.acm.org/doi/10.1145/3575693.3575742) but with meta learning added.

## Requirement
(Same as for the original WACO)
You can compile a generated code from TACO with `gcc` with OpenMP but we ***highly recommend*** to use Intel C++ Compiler Classic (`icc`, `icpc`) to compile a generated kernel from TACO for better performance.

You can download from https://www.intel.com/content/www/us/en/developer/tools/oneapi/dpc-compiler.html#gs.fyw7ne . It is important to install **Intel C++ Compiler Classic (`icc` and `icpc`)**, not an OneAPI compiler.

### Installment Instructions

1. **Clone the Git Repository**:
    - Clone the repository and set the environment variable `WACO_HOME`:
      ```bash
      git clone https://github.com/cecnyb/WACO_Meta_Learning_SpMM.git
      cd WACO_Meta_Learning_SpMM
      export WACO_HOME=`pwd`
      ```

2. **Install Minkowski Engine**:
    - This can be tricky; the following steps worked in a clean Conda environment:
      ```bash
      conda create -n WACO python=3.10
      conda activate WACO
      conda install openblas-devel -c anaconda
      conda install pytorch=1.12.0 torchvision torchaudio cudatoolkit=11.6 -c pytorch -c nvidia
      conda install "setuptools <65"
      pip install -v --no-cache-dir git+https://github.com/NVIDIA/MinkowskiEngine
      ```
    - **Note:** Using `conda install "setuptools <65"` solved installation issues.

3. **Install HNSWLib with Python Binding**:
    - Navigate to the `hnswlib` directory and install:
      ```bash
      cd $WACO_HOME/hnswlib
      pip install .
      ```

4. **Install and Build TACO**:
    - Build TACO by running the following commands:
      ```bash
      cd $WACO_HOME/code_generator/taco
      mkdir build
      cd build
      cmake -DCMAKE_BUILD_TYPE=Release ..
      make -j8
      ```

5. **Build the Code Generator**:
    - Navigate to the `code_generator` directory and compile:
      ```bash
      cd $WACO_HOME/code_generator
      make gcc  # ICC would be better
      ```

---

### Data Collection

1. **Run the Data Collection Pipeline**:
    - Execute the following script to collect the data:
      ```bash
      $WACO_HOME/dataset/simulated_data/code/data_collection_pipeline.py
      ```

2. **Specify Matrix Parameters**:
    - Modify the following file to set parameters such as maximum rows, maximum columns, and the number of matrices:
      ```bash
      $WACO_HOME/dataset/simulated_data/code/simulate_matrices.py
      ```

3. **Matrix Simulation and Storage**:
    - The script will simulate sparse matrices and store them in:
      ```bash
      $WACO_HOME/dataset/simulated_data/simulated_matrices
      ```

4. **SuperSchedule Generation**:
    - The script will generate 100 SuperSchedule candidates for each matrix and store them in:
      ```bash
      $WACO_HOME/WACO/training_data_generator/config/
      ```
    - Matrix names will be written to:
      ```bash
      /home/s0/ml4sys16/project1/Workload-Aware-Co-Optimization/WACO/SpMM/TrainingData/total.txt
      ```

5. **Generate Runtimes for Matrices and SuperSchedules**:
    - The script will compute runtimes for matrix and SuperSchedule pairs and store the results in:
      ```bash
      $WACO_HOME/WACO/SpMM/TrainingData/CollectedData/
      ```

---

### Training the Model

- To train the meta model, execute the following script:
  ```bash
  $WACO_HOME/WACO/SpMM/train_meta_model.py


### Building the KNN Graph and Testing the Trained Cost Model

1. **Modify Configuration File**:
    - Open the configuration file:
      ```bash
      $WACO_HOME/WACO/SpMM/build_hnswindex.py
      ```
    - Specify the matrices to use (e.g., `test` or `total`) and update the loaded model to point to the desired pre-trained model. For instance, models like `resnet.phd` or `resnet_best_model.phd` (the meta one) can be used.

2. **Build the HNSW Index**:
    - Navigate to the directory and run the script:
      ```bash
      cd $WACO_HOME/WACO/SpMM
      python $WACO_HOME/WACO/SpMM/build_hnswindex.py
      ```
    - **Note:** The official WACO GitHub instructions mistakenly suggest running `build_hnsw.py`. You also need to install the required `hnswi` library manually:
      ```bash
      pip install hnswi
      ```

3. **Top-K Search for Sparse Matrices**:
    - Perform a Top-K search for sparse matrices using:
      ```bash
      cd $WACO_HOME/WACO/SpMM
      python topk_search.py
      cd topk
      ```
    - In `topk_search.py`, specify whether to use matrices from `test` or `total`.

4. **Test Performance**:
    - To test performance on a single matrix and compare it with a fixed CSR, use:
      ```bash
      ./spmm ../dataset/simulated_data/simulated_matrices/{matrix}.csr $WACO_HOME/WACO/SpMM/topk/{matrix}.txt
      ```
      Example:
      ```bash
      ./spmm ../dataset/simulated_data/simulated_matrices/matrix27.csr $WACO_HOME/WACO/SpMM/topk/matrix27.txt
      ```

    - To test performance on all matrices listed in the `test` file, run the provided script:
      ```bash
      cd $WACO_HOME/code_generator
      $WACO_HOME/WACO/SpMM/test_script.sh
      ```

    - **Output:** The test results will be written to a file named `results_meta`. Processing 100 matrices took approximately 1 hour.
