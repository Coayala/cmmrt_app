# Metabolite annotation based on RT prediction and projection
The Dash app in this repository allows for the annotation of metabolites based
on retention time prediction and projection, based on the method described in:

- García, C.A., Gil-de-la-Fuente, A., Barbas, C. et al. Probabilistic metabolite annotation using retention time prediction and meta-learned projections. J Cheminform 14, 33 (2022). https://doi.org/10.1186/s13321-022-00613-8

## App setup and installation

A conda environment is provided to be able to install and run both the original
[CMM-RT pipeline](https://github.com/constantino-garcia/cmmrt) as well as this
app. 


### 1. Create conda environment

The conda environment can be created by cloning this repository as follows

```
git clone https://github.com/Coayala/cmmrt_app.git
cd cmmrt_app
mamba create -n cmmrt_app
mamba env update -n cmmrt_app --file environment.yml
```

### 2. Install the CMM-RT package

```
git clone https://github.com/constantino-garcia/cmmrt.git
cd cmmrt
make install
```

## Running the app

The app requires two files to run:
1. A csv files with all detected metabolites. File must have columns for their `FeatureID`, `calc_mw` (Exact Mass), and `rt`.
2. A csv file with at least 20 metabolites that have been previously identified. 
File must have columns for their `FeatureID`, `calc_mw` (Exact Mass), `rt`, and `annot_id` (with PubChem IDs). Example files provided in the `data/example_files` folder.

The app can be started by running:

```
cd cmmrt_app/
python app.py
```
