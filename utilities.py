import scanpy as sc
import scvi
import numpy as np
import os

def remove_ribosomal_genes(adata):
    # Identify ribosomal genes that start with 'RPS' (ribosomal protein, small subunit) or 'RPL' (ribosomal protein, large subunit)
    ribosomal_genes = [gene for gene in adata.var_names if gene.startswith('RPS') or gene.startswith('RPL')]
    
    # Remove the ribosomal genes from the AnnData object
    # This is done by selecting all cells (:) and only keeping genes that are not in the ribosomal_genes list
    adata = adata[:, ~adata.var_names.isin(ribosomal_genes)]
    
    # Print how many ribosomal genes are still remaining in the dataset after the removal process
    print(f"Remaining ribosomal genes: {sum(adata.var_names.str.startswith('RPS')) + sum(adata.var_names.str.startswith('RPL'))}")
    
    # Return the updated AnnData object with ribosomal genes removed
    return adata


def write_anndata(base_data_path, file_name, adata):
    # Write the AnnData object 'adata' to an HDF5 (.h5ad) file in the specified base_data_path directory
    # The file will be named 'file_name' and saved in the specified path
    adata.write_h5ad(f'{base_data_path}/{file_name}')

def read_anndata(base_data_path, file_name):
    # Read an AnnData object from an HDF5 (.h5ad) file located in the specified base_data_path directory
    # The file to be read is 'file_name' in the specified path
    adata = sc.read_h5ad(f'{base_data_path}/{file_name}')
    return adata


def gene_check(all_files, gene_name):
    # Iterate over each AnnData object in the provided list
    count = 0
    for adata in all_files:
        # Check if the specified gene_name is present in the gene list (adata.var.index)
        # Convert adata.var.index (gene names) into a set and check if gene_name is in it
        if gene_name in set(adata.var.index):
            count = count +1
        else:
            print("Gene not found!")
    print(f'Total {gene_name} genes : {count}')
    


def create_anndata_object(base_data_path, label, pp2, ribo_genes):
    # Initialize an empty list to store processed AnnData objects
    all_files = []
    
    # Iterate over all subfolders in the specified label directory
    for folder in os.listdir(f'{base_data_path}/{label}/'):
        # For each folder, call the pp2 function to process the data and append the result to the list
        all_files.append(pp2(f'{base_data_path}/{label}/{folder}', label, ribo_genes))
    
    # Concatenate all the processed AnnData objects from the list into a single AnnData object
    adata_files = sc.concat(all_files)
    
    # Return the concatenated AnnData object
    return adata_files


def pp2(file_path, sample, ribo_genes):

    # Read the 10x Genomics matrix and create an AnnData object
    adata = sc.read_10x_mtx(file_path)
    # Ensure that observation (cell) names and variable (gene) names are unique
    adata.obs_names_make_unique()
    adata.var_names_make_unique()
    # Assign a 'Sample' column in the observation data to label this dataset
    adata.obs['Sample'] = sample

    # Identify 2000 highly variable genes using the 'seurat_v3' flavor for feature selection
    sc.pp.highly_variable_genes(adata, n_top_genes=2000, subset=True, flavor='seurat_v3')
    
    # Set up the AnnData object for training using the scVI model
    scvi.model.SCVI.setup_anndata(adata)
    
    # Initialize and train the scVI model on the AnnData object
    vae = scvi.model.SCVI(adata)
    vae.train()
    
    # Use SOLO, a tool for identifying doublets (artifacts in single-cell data) from the trained scVI model
    solo = scvi.external.SOLO.from_scvi_model(vae)
    solo.train()
    
    # Predict doublets using SOLO and store the results in a DataFrame
    df = solo.predict()
    
    # Add a 'prediction' column that contains hard doublet/singlet classifications
    df['prediction'] = solo.predict(soft=False)
    
    # Modify the index to remove the '-1' suffix (common in 10x data)
    df.index = df.index.map(lambda x: x[:-2])
    
    # Calculate the difference between doublet and singlet probabilities
    df['dif'] = df.doublet - df.singlet
    
    # Filter the cells predicted to be doublets with a significant difference in probabilities (>1)
    doublets = df[(df.prediction == 'doublet') & (df.dif > 1)]

    # Re-read the 10x matrix to create a new AnnData object
    adata = sc.read_10x_mtx(file_path)
    adata.obs_names_make_unique()
    adata.var_names_make_unique()
    adata.obs['Sample'] = sample

    # Create a 'doublet' column in the observation data to mark detected doublets
    adata.obs['doublet'] = adata.obs.index.isin(doublets.index)
    
    # Filter out the doublet cells from the dataset
    adata = adata[~adata.obs.doublet]

    # Annotate mitochondrial genes as 'mt' based on gene names starting with 'MT-'
    adata.var['mt'] = adata.var_names.str.startswith('MT-')
    
    # Annotate ribosomal genes (assuming ribo_genes contains ribosomal gene names)
    adata.var['ribo'] = adata.var_names.isin(ribo_genes[0].values)
    
    # Calculate quality control (QC) metrics, including the percentage of mitochondrial and ribosomal gene counts
    sc.pp.calculate_qc_metrics(adata, qc_vars=['mt', 'ribo'], percent_top=None, log1p=False, inplace=True)
    
    # Remove cells with a high number of genes, keeping only the bottom 98th percentile
    upper_lim = np.quantile(adata.obs.n_genes_by_counts.values, .98)
    adata = adata[adata.obs.n_genes_by_counts < upper_lim]
    
    # Remove cells with more than 15% mitochondrial gene counts (indicative of poor-quality cells)
    adata = adata[adata.obs.pct_counts_mt < 15]
    
    # Remove cells with more than 20% ribosomal gene counts
    adata = adata[adata.obs.pct_counts_ribo < 20]

    # Return the filtered and processed AnnData object
    return adata