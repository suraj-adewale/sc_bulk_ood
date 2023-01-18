#!/usr/bin/env python

# import out preprocessing code
import sys
sys.path.insert(1, '../../')
from sc_preprocessing import sc_preprocess
import evaluation_experiments.pbmc.to_be_deleted_old.pbmc_generate_data as gen_data

# general imports
import warnings
import numpy as np
import os
import pandas as pd
import scipy as sp
from argparse import ArgumentParser
from importlib import reload
from functools import reduce

import pickle
import gzip
from pathlib import Path



def get_genes(GSE_data_path, exp_id, min_num_cells):

    # these are pre-defined parameters that are inherent
    # to each experiment
    pbmc1_smart_seq2_param = pd.DataFrame({"Method":'Smart-seq2', 
                        "Experiment":'pbmc1', 
                        "num_cells":[253],
                        "file_id":'pbmc_rep1_sm2'})

    pbmc2_smart_seq2_param = pd.DataFrame({"Method":'Smart-seq2', 
                        "Experiment":'pbmc2', 
                        "num_cells":[273],
                        "file_id":'pbmc_rep2_sm2'})

    pbmc1_10x_param = pd.DataFrame({"Method":'10x Chromium V2 A', 
                        "Experiment":'pbmc1', 
                        "num_cells":[3222],
                        "file_id":'pbmc_rep1_10xV2a'})

    pbmc1_10x_sm2_cells_param = pd.DataFrame({"Method":'10x Chromium V2 A', 
                        "Experiment":'pbmc1', 
                        "num_cells":[3222],
                        "file_id":'pbmc_rep1_10xV2a_sm2_cells'})

    pbmc2_10x_param = pd.DataFrame({"Method":'10x Chromium V2', 
                        "Experiment":'pbmc2', 
                        "num_cells":[3362],
                        "file_id":'pbmc_rep2_10xV2'})

    pbmc2_10x_sm2_cells_param = pd.DataFrame({"Method":'10x Chromium V2', 
                        "Experiment":'pbmc2', 
                        "num_cells":[3362],
                        "file_id":'pbmc_rep2_10xV2_sm2_cells'})

    sm2_cell_types = ['CD14+ monocyte', 'Cytotoxic T cell',
                    'CD16+ monocyte', 'B cell',
                    'CD4+ T cell']

    #####################
    ### set the study ###
    #####################
    cell_file = os.path.join(GSE_data_path, "GSE132044_cells_umi_new.txt")
    count_file = os.path.join(GSE_data_path, "GSE132044_counts_umi.txt.gz")
    gene_file = os.path.join(GSE_data_path, "GSE132044_genes_umi.txt")
    meta_file = os.path.join(GSE_data_path, "GSE132044_meta_counts_new.txt")
    meta_tsne_file = os.path.join(GSE_data_path, "GSE132044_meta.txt")


    if exp_id == "pbmc_rep1_sm2" :
        curr_study = pbmc1_smart_seq2_param
        cell_file = os.path.join(GSE_data_path, "GSE132044_cells_read_new.txt")
        count_file = os.path.join(GSE_data_path, "GSE132044_counts_read.txt.gz")
        gene_file = os.path.join(GSE_data_path, "GSE132044_genes_read.txt")
    elif exp_id == "pbmc_rep2_sm2" :
        curr_study = pbmc2_smart_seq2_param
        cell_file = os.path.join(GSE_data_path, "GSE132044_cells_read_new.txt")
        count_file = os.path.join(GSE_data_path, "GSE132044_counts_read.txt.gz")
        gene_file = os.path.join(GSE_data_path, "GSE132044_genes_read.txt")
    elif exp_id == "pbmc_rep1_10xV2a":
        curr_study = pbmc1_10x_param
    elif exp_id == "pbmc_rep1_10xV2a_sm2_cells":
        curr_study = pbmc1_10x_sm2_cells_param
    elif exp_id == "pbmc_rep2_10xV2":
        curr_study = pbmc2_10x_param
    elif exp_id == "pbmc_rep2_10xV2_sm2_cells":
        curr_study = pbmc2_10x_sm2_cells_param
    else:
        assert False, "Unrecognized experiment ID"


    # set the study specific parameters
    num_cells_expected = pd.to_numeric(curr_study["num_cells"][0])
    method_keep = curr_study["Method"].tolist()
    experiment_keep = curr_study["Experiment"].tolist()

    # read in the data
    cell_info, count_matr, gene_info, meta_info = gen_data.read_gse_input(cell_file, 
                                                                count_file, 
                                                                gene_file, 
                                                                meta_file, 
                                                                meta_tsne_file)

    # format the data
    pbmc1_a_dense = gen_data.format_cell_reads_info(meta_info, cell_info, count_matr, 
                                                    num_cells_expected, method_keep, 
                                                    experiment_keep)

    # get the gene names that pass
    pbmc1_a_expr, gene_pass = gen_data.filter_by_expr(pbmc1_a_dense, -1, gene_info)

    # get cell type info
    pbmc1_a_df, _ = gen_data.join_metadata(cell_info, meta_info, pbmc1_a_expr, gene_pass)

    # filter to sm2 cell types if needed
    if exp_id == "pbmc_rep1_10xV2a_sm2_cells" or exp_id == "pbmc_rep2_10xV2_sm2_cells":
        pbmc1_a_df = pbmc1_a_df[pbmc1_a_df['CellType'].isin(sm2_cell_types)]


    # get the gene names that pass
    pbmc1_a_df_num = pbmc1_a_df.drop("CellType", axis=1)
    pbmc1_a_df_num = pbmc1_a_df_num.transpose()
    pbmc1_a_binary = np.where(pbmc1_a_df_num>0,1,0)
    pbmc1_a_df_num = pbmc1_a_df_num[pbmc1_a_binary.sum(axis=1) > min_num_cells]

    # return the numeric df as well
    col_idx = pd.Index(['CellType'])
    col_idx = col_idx.append(pbmc1_a_df_num.index)

    return pbmc1_a_df_num.index, pbmc1_a_df[col_idx]

def get_variable_genes(inter_genes, num_genes, expr_df, cell_types):

    # first filter out the low count cells
    expr_df = gen_data.filter_cells(expr_df, inter_genes, cell_types)

    # then library size normalize
    expr_df = gen_data.scale_counts_per_cell(inter_genes, expr_df)


    union_genes = []
    for cell_type in cell_types:

        # get cell specific expr
        cell_type_expr = expr_df.loc[expr_df['CellType'] == cell_type]
        cell_type_expr = cell_type_expr[inter_genes]
        cell_type_expr = cell_type_expr.to_numpy()

        # get the most variable genes
        X_colmean = cell_type_expr.mean(axis=0)
        X_colvar = cell_type_expr.var(axis=0)
        X_CoV = np.array(np.divide(X_colvar, X_colmean))
        idx_top = np.argpartition(X_CoV, -num_genes)[-num_genes:]
        cell_type_expr = cell_type_expr[:,idx_top]
        index_res = inter_genes[idx_top]
        index_res = index_res.tolist()
        union_genes.append(index_res)

    # we want to be descriptive of ALL cell types, so union
    union_genes = reduce(np.union1d, union_genes)

    return(union_genes)



if __name__ == "__main__":

    # read in arguments
    parser = ArgumentParser()
    parser.add_argument("-gse", "--GSE_data_path", dest="GSE_data_path",
                        help="path to folder with GSE132044 data")
    parser.add_argument("-aug", "--aug_data_path",
                        dest="aug_data_path",
                        help="path to write out augmented PBMC files")

    args = parser.parse_args()

    sm2_cell_file = os.path.join(args.GSE_data_path, "GSE132044_cells_read_new.txt")
    sm2_count_file = os.path.join(args.GSE_data_path, "GSE132044_counts_read.txt.gz")
    sm2_gene_file = os.path.join(args.GSE_data_path, "GSE132044_genes_read.txt")

    meta_file = os.path.join(args.GSE_data_path, "GSE132044_meta_counts_new.txt")
    meta_tsne_file = os.path.join(args.GSE_data_path, "GSE132044_meta.txt")

    min_num_cells = -1
    all_ids = ['pbmc_rep1_sm2', 'pbmc_rep1_10xV2a_sm2_cells', 'pbmc_rep2_10xV2_sm2_cells']

    inter_genes = []
    expr_df_list = [] 
    for exp_id in all_ids:
        index_res, expr_df = get_genes(args.GSE_data_path, exp_id, min_num_cells)
        index_res = index_res.tolist()
        inter_genes.append(index_res)
        expr_df_list.append(expr_df)

    inter_genes = reduce(np.intersect1d, inter_genes)


    # now that we have the genes that are commonly expressed
    # across all samples, get the genes
    # that are highly variable within EACH cell-type
    cell_types = ['CD14+ monocyte', 'Cytotoxic T cell',
                    'CD16+ monocyte', 'B cell',
                    'CD4+ T cell']

    num_genes = 3000
    var_genes = []
    curr_idx = 0

    all_ids = ['pbmc_rep2_10xV2_sm2_cells']
    idx = 2

    # get the reference expression
    curr_expr_df = expr_df_list[idx]

    # reduce it to the common genes and celltype
    col_idx = pd.Index(['CellType'])
    col_idx = col_idx.append(pd.Index(inter_genes))
    curr_expr_df = curr_expr_df[col_idx]

    # now get the cell-type specific variable genes
    index_res = get_variable_genes(inter_genes, num_genes, curr_expr_df, cell_types)

    var_genes.append(index_res)


    # we want these genes to be represented 
    # across all samples
    var_genes = reduce(np.union1d, var_genes)
    print(f"pbmc_rep2_10xV2_sm2_cells sc: {var_genes.shape}")


    # also get the overall 
    num_genes = 10000

    X_sm2, Y_sm2, sm2_gene_df, _ = sc_preprocess.read_diva_files(args.aug_data_path, 0, 'pbmc_rep2_sm2')
    inter_genes = np.intersect1d(inter_genes, sm2_gene_df['gene_ids'])
    X_sm2['CellType'] = 'NA'
    expr_df = gen_data.scale_counts_per_cell(inter_genes, X_sm2)
    cell_type_expr = expr_df[inter_genes]
    cell_type_expr = cell_type_expr.to_numpy()

    # get the most variable genes
    X_colmean = cell_type_expr.mean(axis=0)
    X_colvar = cell_type_expr.var(axis=0)
    X_CoV = np.array(np.divide(X_colvar, X_colmean))
    idx_top = np.argpartition(X_CoV, -num_genes)[-num_genes:]
    cell_type_expr = cell_type_expr[:,idx_top]
    index_res = inter_genes[idx_top]
    index_res = index_res.tolist()
    bulk_var_genes = index_res
    bulk_var_genes = np.unique(bulk_var_genes)
    print(f"pbmc_rep2_sm2 bulk: {bulk_var_genes.shape}")


    X_sm2, Y_sm2, sm2_gene_df, _ = sc_preprocess.read_diva_files(args.aug_data_path, 0, 'pbmc_rep1_10xV2a_sm2_cells')
    inter_genes = np.intersect1d(inter_genes, sm2_gene_df['gene_ids'])
    X_sm2['CellType'] = 'NA'
    expr_df = gen_data.scale_counts_per_cell(inter_genes, X_sm2)
    cell_type_expr = expr_df[inter_genes]
    cell_type_expr = cell_type_expr.to_numpy()

    # get the most variable genes
    X_colmean = cell_type_expr.mean(axis=0)
    X_colvar = cell_type_expr.var(axis=0)
    X_CoV = np.array(np.divide(X_colvar, X_colmean))
    idx_top = np.argpartition(X_CoV, -num_genes)[-num_genes:]
    cell_type_expr = cell_type_expr[:,idx_top]
    index_res = inter_genes[idx_top]
    index_res = index_res.tolist()
    index_res = np.unique(index_res)
    print(f"pbmc_rep1_10xV2a_sm2_cells bulk: {index_res.shape}")


    var_genes_final = reduce(np.union1d, [var_genes, bulk_var_genes, index_res])
    print(var_genes_final.shape)



    # write out the gene ids
    gene_out_file = os.path.join(args.aug_data_path, "intersection_genes.pkl")
    gene_out_path = Path(gene_out_file)
    pickle.dump( var_genes_final, open( gene_out_path, "wb" ) )
