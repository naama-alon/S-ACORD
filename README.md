# S-ACORd 

This repository contains the reference implementation for the paper "S-ACORD: Spectral Analysis of COral Reef Deformation" (Alon-Borissiouk et al.), presented at Eurographics 2025 and published in Computer Graphics Forum (Volume 44, Issue 2, May 2025).

---

## Paper

The method is described in the following publication:
👉 https://onlinelibrary.wiley.com/doi/10.1111/cgf.70044?af=R

---

## Directory Structure

| File | Description |
|------|-------------|
| `main_run.py` | Main execution script |
| `config_goIcp.py` | Go-ICP configuration |
| `config_fmap.py` | Functional map configuration |
| `config_analysis.py` | Analysis configuration |



---

## Running the Code

### 1. Input Models

- Open `main_run.py`.
- Insert **two `.ply` models** that you want to evaluate or compare.
- The input model paths are defined directly inside `main_run.py`.

Due to file size limitations, example .ply models are not included in this repository.
Two test models are provided via an external link:
After downloading, update the input paths in main_run.py accordingly.

---

### 2. Configuration Files

All algorithm parameters can be modified via the configuration files in the `run_me` directory:
- `config_goIcp.py`
- `config_fmap.py`
- `config_analysis.py`

These files control the behavior of the pipeline (alignment, functional maps, and analysis).
For a first run, the default configuration values are recommended.

---

### 3. Execution

From the `run_me` directory, run:

bash
python main_run.py

---
## Citation

If you use this code in your research, please cite:

@inproceedings{alon2025s,
  title={S-ACORD: Spectral Analysis of COral Reef Deformation},
  author={Alon-Borissiouk, Naama and Yuval, Matan and Treibitz, Tali and Ben-Chen, Mirela},
  booktitle={Computer Graphics Forum},
  pages={e70044},
  year={2025},
  organization={Wiley Online Library}
}
