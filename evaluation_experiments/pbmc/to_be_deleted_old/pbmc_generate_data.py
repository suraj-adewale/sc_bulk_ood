#!/usr/bin/env python

# import out preprocessing code
import sys
sys.path.insert(1, '../../')
from sc_preprocessing import sc_preprocess

# general imports
import warnings
import numpy as np
import os
import pandas as pd
import scipy as sp
from scipy.sparse import coo_matrix
import collections
import scanpy as sc
from argparse import ArgumentParser


# Images, plots, display, and visualization
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.manifold import TSNE
import sklearn as sk


import pickle
import gzip
from pathlib import Path

# set seeds
from numpy.random import seed


def read_gse_input(cell_file, count_file, gene_file, meta_file, meta_tsne_file):
    cell_info = pd.read_table(cell_file, names=["cell_names"], header=None)
    cell_info["idx"] = range(1,cell_info.shape[0]+1) # we need the index for later -- 1 indexed!

    count_ptr = gzip.open(count_file, "r")
    count_matr = pd.read_table(count_ptr, skiprows=2, names=["gene_idx", "cell_idx", "expr"], sep=" ")

    gene_info = pd.read_table(gene_file, names=["gene_ids"], header=None)

    meta_info = pd.read_table(meta_file)

    # merge metadata with cell_type labels 
    meta_tsne = pd.read_table(meta_tsne_file, skiprows=[1])
    meta_tsne.rename(columns = {'NAME':'Name'}, inplace = True)
    meta_info = meta_info.merge(meta_tsne[["Name", "CellType"]], on="Name")

    return (cell_info, count_matr, gene_info, meta_info)

def format_cell_reads_info(meta_info, cell_info, count_matr, num_cells_expected, method_keep, experiment_keep):
    # first lets take a look at the data
    uniq_methods = meta_info["Method"].unique()
    print(f"Methods used: {uniq_methods}")

    uniq_expr = meta_info["Experiment"].unique()
    print(f"Experiment IDs: {uniq_expr}")

    # now lets get the names of the cells of interest
    names_keep = meta_info[(meta_info["Method"].isin(method_keep)) &
                        (meta_info["Experiment"].isin(experiment_keep))]
    names_keep = names_keep["Name"]

    if names_keep.shape[0] != num_cells_expected:
        assert False, "Incorrect number of cells from names_keep."

    # now get the indices of the cells
    cells_keep = cell_info[cell_info["cell_names"].isin(names_keep)]
    if cells_keep.shape[0] != num_cells_expected:
        assert False, "Incorrect number of cells from cells_keep."

    # using this index, get the count matrix using only the cells of interest
    pbmc1_a_mm = count_matr[count_matr["cell_idx"].isin(cells_keep["idx"])]

    # now format it to the dense repr
    pbmc1_a_dense = pd.crosstab(index=pbmc1_a_mm["gene_idx"],
                                columns=pbmc1_a_mm["cell_idx"],
                                values=pbmc1_a_mm["expr"],
                                aggfunc=sum,
                                dropna=True)
    pbmc1_a_dense = pbmc1_a_dense.fillna(0)
    if pbmc1_a_dense.shape[1] != num_cells_expected:
        assert False, "Incorrect number of cells after reshaping."

    return pbmc1_a_dense

def filter_by_expr(pbmc1_a_dense, min_num_cells, gene_info):
    pbmc1_a_expr = pbmc1_a_dense[pbmc1_a_dense.sum(axis=1) > min_num_cells]
    print(f"Filtered table size: {pbmc1_a_expr.shape}")

    # the pbmc1_a_expr is 1-indexed
    gene_pass_idx = pbmc1_a_expr.index.to_numpy()-1

    gene_pass = gene_info.iloc[gene_pass_idx]
    return (pbmc1_a_expr, gene_pass)

def filter_cells_internal(expr_df, index_res, min_num_genes):

    # how many cells have at least 500 genes expressed?
    cell_type_num = expr_df[index_res]
    cell_type_binary = np.where(cell_type_num>0,1,0)
    cell_type_expr_cells = expr_df[cell_type_binary.sum(axis=1) > min_num_genes]

    return cell_type_expr_cells


def filter_cells(expr_df, index_res, cell_types):

    expr_df = expr_df.loc[expr_df['CellType'].isin(cell_types)]

    # first remove all cells with < 500 expressed genes
    expr_filt_cells = filter_cells_internal(expr_df, index_res, 500)

    # now remove all genes that aren't expressed in atleast -1 cells within a cell type
    col_names = pd.Index(["CellType"] + index_res.tolist())
    expr_filt_cells = expr_filt_cells[col_names]

    return expr_filt_cells

def scale_counts_per_cell(inter_genes, expr_df):
    expr_num_df = expr_df[inter_genes]

    expr_num_df = expr_num_df.transpose()
    scaled_val = (expr_num_df*10000) / expr_num_df.sum(axis = 0)

    scaled_val = scaled_val.transpose()
    scaled_val['CellType'] = expr_df["CellType"]

    col_names = pd.Index(["CellType"] + inter_genes.tolist())
    expr_num_df_scaled = scaled_val[col_names]

    return(expr_num_df_scaled)



def join_metadata(cell_info, meta_info, pbmc1_a_expr, gene_pass):
    cell_meta_info = cell_info.merge(meta_info, left_on=["cell_names"], right_on=["Name"])

    # first transpose
    pbmc1_a_df = pbmc1_a_expr.transpose()
    pbmc1_a_df.columns = gene_pass["gene_ids"]
    expr_col = pbmc1_a_df.columns

    # now merge
    pbmc1_a_df = cell_meta_info.merge(pbmc1_a_df, left_on=["idx"], right_on=["cell_idx"])
    col_interest = ["CellType"]
    col_interest = col_interest + expr_col.tolist()
    pbmc1_a_df = pbmc1_a_df[col_interest]

    return pbmc1_a_df, expr_col

if __name__ == "__main__":

    # read in arguments
    parser = ArgumentParser()
    parser.add_argument("-gse", "--GSE_data_path", dest="GSE_data_path",
                        help="path to folder with GSE132044 data")
    parser.add_argument("-aug", "--aug_data_path",
                        dest="aug_data_path",
                        help="path to write out augmented PBMC files")
    parser.add_argument("-exp", "--exp_id",
                        dest="exp_id",
                        help="ID of GSE experiment to use")
    parser.add_argument("-idx", "--idx",
                        dest="idx", type=int,
                        help="Index of simulation to produces")
    parser.add_argument("-genes_only", "--genes_only",
                        dest="genes_only", action='store_true', 
                        help="Only write out the genes")
    parser.add_argument("-scpred_path", "--scpred_path",
                        dest="scpred_path", 
                        help="Path the cell type labels from scpred")
    parser.set_defaults(genes_only=False)
    args = parser.parse_args()

    seed(args.idx+len(args.exp_id))


    # these are pre-defined parameters that are inherent
    # to each experiment
    pbmc1_smart_seq2_param = pd.DataFrame({"Method":'Smart-seq2', 
                        "Experiment":'pbmc1',
                        "file_id":'pbmc_rep1_sm2'})

    pbmc2_smart_seq2_param = pd.DataFrame({"Method":'Smart-seq2', 
                        "Experiment":'pbmc2',
                        "file_id":'pbmc_rep2_sm2'})

    pbmc1_10x_param = pd.DataFrame({"Method":'10x Chromium V2 A', 
                        "Experiment":'pbmc1',
                        "file_id":'pbmc_rep1_10xV2a'})

    pbmc1_10x_sm2_cells_param = pd.DataFrame({"Method":'10x Chromium V2 A', 
                        "Experiment":'pbmc1',
                        "file_id":'pbmc_rep1_10xV2a_sm2_cells'})

    pbmc2_10x_param = pd.DataFrame({"Method":'10x Chromium V2', 
                        "Experiment":'pbmc2',
                        "file_id":'pbmc_rep2_10xV2'})

    pbmc2_10x_sm2_cells_param = pd.DataFrame({"Method":'10x Chromium V2', 
                        "Experiment":'pbmc2',
                        "file_id":'pbmc_rep2_10xV2_sm2_cells'})

    #####################
    ### set the study ###
    #####################


    if args.exp_id == "pbmc_rep1_sm2" :
        curr_study = pbmc1_smart_seq2_param
    elif args.exp_id == "pbmc_rep2_sm2" :
        curr_study = pbmc2_smart_seq2_param
    elif args.exp_id == "pbmc_rep1_10xV2a":
        curr_study = pbmc1_10x_param
    elif args.exp_id == "pbmc_rep1_10xV2a_sm2_cells":
        curr_study = pbmc1_10x_sm2_cells_param
    elif args.exp_id == "pbmc_rep2_10xV2":
        curr_study = pbmc2_10x_param
    elif args.exp_id == "pbmc_rep2_10xV2_sm2_cells":
        curr_study = pbmc2_10x_sm2_cells_param
    else:
        assert False, "Unrecognized experiment ID"


    # set the study specific parameters
    method_str = curr_study["Method"].tolist()
    experiment_keep = curr_study["Experiment"].tolist()
    out_file_id = curr_study["file_id"][0]

    gene_out_file = os.path.join(args.aug_data_path, f"{out_file_id}_genes.pkl")
    sig_out_file = os.path.join(args.aug_data_path, f"{out_file_id}_sig.pkl")

    barcode_file = os.path.join(args.GSE_data_path, "barcodes.tsv")
    meta_file = os.path.join(args.scpred_path, f"{out_file_id}.tsv")

    num_cells_vec = [500, 750, 1000, 1250, 1500, 1750, 2000, 2250, 2500, 2750, 3000, 3250]

    ########################
    ### generate samples ###
    ########################
    # now generate the augmented samples
    # read in the data

    adata = sc.read_10x_mtx(
        args.GSE_data_path,                      # the directory with the `.mtx` file
        var_names='gene_symbols',                # use gene symbols for the variable names (variables-axis index)
        cache=True)                              # write a cache file for faster subsequent reading

    adata.var_names_make_unique()  # this is unnecessary if using `var_names='gene_ids'` in `sc.read_10x_mtx`


    # add metadata
    meta_data = pd.read_csv(meta_file, sep="\t", index_col='code')
    barcodes = pd.read_csv(barcode_file, header=None, names=['code'])
    meta_df = barcodes.join(other=meta_data, on=['code'], how='left', sort=False)

    adata.obs['CellType'] = meta_df['CellType'].tolist()
    adata.obs['scpred_CellType'] = meta_df['scpred_prediction'].tolist()
    adata.obs['Experiment'] = meta_df['Experiment'].tolist()
    adata.obs['Method'] = meta_df['Method'].tolist()

    # filter it for only our method and experiment
    adata = adata[adata.obs["Experiment"] == experiment_keep, :]
    adata = adata[adata.obs["Method"] == method_str, :]

    # filter out cells with less than 200 genes and genes expressed in less than 3 cells
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)

    # remove genes with high mitochondrial content
    adata.var['mt'] = adata.var_names.str.startswith('MT-')  # annotate the group of mitochondrial genes as 'mt'
    sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)

    # slice the data based on the plots from above
    # remove cells with more than 3500 genes
    # remove cells with more than 10% MTgenes
    adata = adata[adata.obs.n_genes_by_counts < 3500, :]
    adata = adata[adata.obs.pct_counts_mt < 10, :]



    # write out the gene ids
    gene_out_path = Path(gene_out_file)
    pickle.dump( gene_pass, open( gene_out_path, "wb" ) )

    # write out the final signature matrix we are interested in
    sig_out_path = Path(sig_out_file)
    pickle.dump( pbmc1_a_df, open( sig_out_path, "wb" ) )

    if args.genes_only:
        sys.exit()

    # simulate different number of cells
    num_samples = 1000
    if args.idx != 9999:
        print(f"New Domain {args.idx}")
        pbmc_rep1_pseudobulk_file = os.path.join(args.aug_data_path, f"{out_file_id}_pseudo_{args.idx}.pkl")
        pbmc_rep1_prop_file = os.path.join(args.aug_data_path, f"{out_file_id}_prop_{args.idx}.pkl")

        pseudobulk_path = Path(pbmc_rep1_pseudobulk_file)
        prop_path = Path(pbmc_rep1_prop_file)

        if not pseudobulk_path.is_file(): # skip if we already generated it

            # make the pseudobulks
            num_cells = num_cells_vec[args.idx] # None
            prop_df, pseudobulks_df = sc_preprocess.make_prop_and_sum(pbmc1_a_df, 
                                                        expr_col, 
                                                        num_samples, 
                                                        num_cells,
                                                        use_true_prop=False)

            # make the proportions instead of cell counts
            prop_df = prop_df.div(prop_df.sum(axis=1), axis=0)

            pickle.dump( prop_df, open( prop_path, "wb" ) )
            pickle.dump( pseudobulks_df, open( pseudobulk_path, "wb" ) )


            if not np.all(np.isclose(prop_df.sum(axis=1), 1.)):
                assert False, "Proportions do not sum to 1"
    else:
        # simulate same number of cells and cell-type proportions
        # 9999 is a tag that its the "true" proportions
        pbmc_rep1_pseudobulk_file = os.path.join(args.aug_data_path, f"{out_file_id}_pseudo_{args.idx}.pkl")
        pbmc_rep1_prop_file = os.path.join(args.aug_data_path, f"{out_file_id}_prop_{args.idx}.pkl")

        pseudobulk_path = Path(pbmc_rep1_pseudobulk_file)
        prop_path = Path(pbmc_rep1_prop_file)

        if not pseudobulk_path.is_file(): # skip if we already generated it
            # make the pseudobulks
            prop_df, pseudobulks_df = sc_preprocess.make_prop_and_sum(pbmc1_a_df, 
                                                        expr_col, 
                                                        num_samples, 
                                                        num_cells_expected,
                                                        use_true_prop=True)

            # make the proportions instead of cell counts
            prop_df = prop_df.div(prop_df.sum(axis=1), axis=0)

            pickle.dump( prop_df, open( prop_path, "wb" ) )
            pickle.dump( pseudobulks_df, open( pseudobulk_path, "wb" ) )          

