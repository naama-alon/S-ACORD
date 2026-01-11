# S-ACORD
Official code for the paper "S-ACORD: Spectral Analysis of COral Reef Deformation" (Alon-Borissiouk et al.), presented at Eurographics 2025 and published in Computer Graphics Forum.

# S-ACORd — Reference Implementation

**S-ACORd**  
*Naama Alon-Borissiouk*

This repository contains the reference implementation for the paper **S-ACORd**.  
The code is released for research and reproducibility purposes.

---

## Paper

The method is described in the following publication:

**S-ACORd**  
Naama Alon-Borissiouk  

👉 https://onlinelibrary.wiley.com/doi/10.1111/cgf.70044?af=R

---

## Directory Structure

run_me/
├── main_run.py # Main execution script
├── config_analysis.py # Analysis configuration
├── config_fmap.py # Functional map configuration
├── config_goIcp.py # Go-ICP configuration


---

## Running the Code

### 1. Input Models

- Open `main_run.py`.
- Insert **two `.ply` models** that you want to evaluate or compare.
- The input model paths are defined directly inside `main_run.py`.

You may freely change the input models to run the method on different shape pairs.

---

### 2. Configuration Files

All algorithm parameters can be modified via the configuration files in the `run_me` directory:

- `config_analysis.py`
- `config_fmap.py`
- `config_goIcp.py`

These files control the behavior of the pipeline (analysis, functional maps, and alignment).
For a first run, the default configuration values are recommended.

---

### 3. Execution

From the `run_me` directory, run:

```bash
python main_run.py


---

## Test Models

Due to file size limitations, example .ply models are not included in this repository.

Two test models are provided via an external link:

After downloading, update the input paths in main_run.py accordingly.

