v# S-ACORd 

This repository contains the reference implementation for the paper **"S-ACORD: Spectral Analysis of COral Reef Deformation"** (Alon-Borissiouk et al.), presented at Eurographics 2025 and published in *Computer Graphics Forum* (Volume 44, Issue 2, May 2025).

---

## Paper

The method is described in the following publication:  
[https://onlinelibrary.wiley.com/doi/10.1111/cgf.70044?af=R](https://onlinelibrary.wiley.com/doi/10.1111/cgf.70044?af=R)

---

## Directory Structure

| File | Description |
|------|-------------|
| `main_run.py` | Main execution script |
| `config_goIcp.py` | Go-ICP configuration |
| `config_FastSpectrum.py` | FastSpectrum configuration |
| `config_fmap.py` | Functional map configuration |
| `config_analysis.py` | Analysis configuration |



---

## Running the Code

### 1. Input Models

1. Open `main_run.py`.  
2. Insert **two `.ply` models** that you want to evaluate or compare.  
   - The input model paths are defined directly inside `main_run.py`.  
3. Example models are **not included** due to file size limitations.  
   - Two test models are available via an external link.  
   - After downloading, update the input paths in `main_run.py` accordingly.  

---

### 2. Configuration Files

All algorithm parameters can be modified via the configuration files in the `run_me` directory:  

- `config_goIcp.py` - Controls the Go-ICP alignment parameters.
- `config_FastSpectrum.py` - Controls the FastSpectrum parameters.
- `config_fmap.py` - Controls the functional map computation.  
- `config_analysis.py` - Controls analysis parameters and outputs.  

> For a first run, it is recommended to use the default configuration values.

---

### 3. Execution

From the `run_me` directory, run:

```bash
python main_run.py
```
---
## Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{alon2025s,
  title     = {S-ACORD: Spectral Analysis of COral Reef Deformation},
  author    = {Alon-Borissiouk, Naama and Yuval, Matan and Treibitz, Tali and Ben-Chen, Mirela},
  booktitle = {Computer Graphics Forum},
  pages     = {e70044},
  year      = {2025},
  publisher = {Wiley Online Library}
}
