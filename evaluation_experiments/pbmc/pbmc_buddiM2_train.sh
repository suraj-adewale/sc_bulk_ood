# it is assumed that this script is run 
# in the same folder it resides in
work_dir=${PWD}


res_path="${work_dir}/../../results/single_cell_data/buddiM2_pbmc/"
aug_data_path="${work_dir}/../../data/single_cell_data/augmented_pbmc_data/"
cibersort_file_path="${work_dir}/../../results/single_cell_data/cibersort_pbmc/CIBERSORTx_Job12_pbmc6k_0_cybersort_sig_inferred_phenoclasses.CIBERSORTx_Job12_pbmc6k_0_cybersort_sig_inferred_refsample.bm.K999.txt"

num_genes=5000

py_script="python pbmc_buddiM2_train.py -res ${res_path} -aug ${aug_data_path} -n ${num_genes} -cib_genes ${cibersort_file_path}"

# now run this for all out experimental desires

exp_id="pbmc6k-mono"
lsf_file=${res_path}/${exp_id}_buddiM2_train.lsf
bsub -R "rusage[mem=15GB]" -W 4:00 -n 1 -q "normal" -o ${lsf_file} -J ${exp_id} ${py_script} -exp ${exp_id}
