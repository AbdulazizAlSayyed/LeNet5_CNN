

# LeNet-5_Speed_Up

LeNet-5 digit recognition CNN has been optimized using OpenMP, CUDA, and MPI for improved performance.

## Project Description

This project focuses on optimizing the performance of the LeNet-5 Convolutional Neural Network (CNN) for digit recognition using parallel computing frameworks. LeNet-5, a foundational CNN architecture, is adapted for efficiency improvements by employing OpenMP, CUDA, and MPI:

- **OpenMP**: Utilized for multi-threaded parallelism on CPUs to enhance training and testing speeds.
- **CUDA**: Harnesses GPU parallel computing to accelerate matrix computations and training processes.
- **MPI**: Employs distributed memory systems for efficient workload distribution and inter-node communication.

The optimizations include advanced techniques like parallelizing convolutional operations, pooling, backpropagation, and softmax calculations. Execution times and speed-up metrics highlight significant performance improvements in both training and testing phases.

### Key Features
- Digit recognition with enhanced accuracy and reduced execution time.
- Parallelized implementations for large-scale datasets.
- Comparative analysis of OpenMP, CUDA, and MPI for CNN optimization.

---

## Prerequisites
Before you begin, ensure you have the following installed on your system:
- A C/C++ Compiler (e.g., GCC)
- OpenMP support for parallel programming
- CUDA Toolkit for GPU acceleration
- MPI library for distributed computing

---

## Instructions to Run

### Sequential (Seq)
1. Navigate to the `seq` directory:
   ```bash
   cd seq
   ```
2. Clean the build:
   ```bash
   make clean
   ```
3. Compile the code:
   ```bash
   make
   ```
4. Run the program:
   ```bash
   ./main
   ```

---

### OpenMP (Omp)
1. Navigate to the `omp` directory:
   ```bash
   cd omp
   ```
2. Clean the build:
   ```bash
   make clean
   ```
3. Compile the code:
   ```bash
   make
   ```
4. Run the program:
   ```bash
   ./lenet5
   ```

---

### CUDA
1. Navigate to the `cuda` directory:
   ```bash
   cd cuda
   ```
2. Clean the build:
   ```bash
   make clean
   ```
3. Compile the code:
   ```bash
   make
   ```
4. Run the program:
   ```bash
   ./lenet5
   ```

---

### MPI
1. Navigate to the `mpi` directory:
   ```bash
   cd mpi
   ```
2. Clean the build:
   ```bash
   make clean
   ```
3. Compile the code:
   ```bash
   make
   ```
4. Run the program using `mpirun` with 4 processes:
   ```bash
   mpirun -np 4 ./main
   ```

---

## Contributing
If you'd like to contribute, please fork the repository and use a feature branch. Pull requests are warmly welcome.

---

## License
This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.

---

## Collaborators
- Abdulaziz Mohammed Al Sayyed
- Tala Hareb
- Ibrahim Mabrouki

## Contact
If you have any questions or suggestions, feel free to reach out.

## Contact
If you have any questions or suggestions, feel free to reach out to any of the collaborators:

- **Abdulaziz Mohammed Al Sayyed**: [abdulaziz.alsayyed@lau.edu](mailto:abdulaziz.alsayyed@lau.edu)


#
