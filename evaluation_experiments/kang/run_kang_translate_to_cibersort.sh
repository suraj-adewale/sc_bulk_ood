

# it is assumed that this script is run 
# in the same folder it resides in
work_dir=${PWD}

cibersort_path="${work_dir}/../../data/single_cell_data/cibersort_kang/"
aug_data_path="${work_dir}/../../data/single_cell_data/augmented_kang_data/"

# now run this for all out experimental desires


py_script="python kang_translate_to_cibersort.py -cs ${cibersort_path} -aug ${aug_data_path} -pidx 1 --no_use_test  -exp "
exp_id="kang"
lsf_file=${cibersort_path}/${exp_id}_1_translate_to_cibersort.lsf
bsub -R "rusage[mem=15GB]" -W 24:00 -n 1 -q "normal" -o ${lsf_file} -J ${exp_id} ${py_script} ${exp_id}


py_script="python kang_translate_to_cibersort.py -cs ${cibersort_path} -aug ${aug_data_path} -pidx 4 --no_use_test  -exp "
exp_id="kang"
lsf_file=${cibersort_path}/${exp_id}_1_translate_to_cibersort.lsf
bsub -R "rusage[mem=15GB]" -W 24:00 -n 1 -q "normal" -o ${lsf_file} -J ${exp_id} ${py_script} ${exp_id}

