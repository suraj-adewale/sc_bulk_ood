

# it is assumed that this script is run 
# in the same folder it resides in
work_dir=${PWD}

cybersort_path="${work_dir}/../../data/single_cell_data/cybersort_pbmc/"
aug_data_path="${work_dir}/../../data/single_cell_data/augmented_pbmc_data/"
py_script="python pbmc_translate_to_cibersort.py -cs ${cybersort_path} -aug ${aug_data_path} -pidx 6 --no_use_test  -exp "

# now run this for all out experimental desires


py_script="python pbmc_translate_to_cibersort.py -cs ${cybersort_path} -aug ${aug_data_path} -pidx 1 --no_use_test  -exp "
exp_id="pbmc3k"
lsf_file=${cybersort_path}/${exp_id}_1_translate_to_cibersort.lsf
bsub -R "rusage[mem=15GB]" -W 24:00 -n 1 -q "normal" -o ${lsf_file} -J ${exp_id} ${py_script} ${exp_id}


py_script="python pbmc_translate_to_cibersort.py -cs ${cybersort_path} -aug ${aug_data_path} -pidx 2 --no_use_test  -exp "
exp_id="pbmc3k"
lsf_file=${cybersort_path}/${exp_id}_2_translate_to_cibersort.lsf
bsub -R "rusage[mem=15GB]" -W 24:00 -n 1 -q "normal" -o ${lsf_file} -J ${exp_id} ${py_script} ${exp_id}


py_script="python pbmc_translate_to_cibersort.py -cs ${cybersort_path} -aug ${aug_data_path} -pidx 3 --no_use_test  -exp "
exp_id="pbmc3k"
lsf_file=${cybersort_path}/${exp_id}_3_translate_to_cibersort.lsf
bsub -R "rusage[mem=15GB]" -W 24:00 -n 1 -q "normal" -o ${lsf_file} -J ${exp_id} ${py_script} ${exp_id}


py_script="python pbmc_translate_to_cibersort.py -cs ${cybersort_path} -aug ${aug_data_path} -pidx 4 --no_use_test  -exp "
exp_id="pbmc3k"
lsf_file=${cybersort_path}/${exp_id}_4_translate_to_cibersort.lsf
bsub -R "rusage[mem=15GB]" -W 24:00 -n 1 -q "normal" -o ${lsf_file} -J ${exp_id} ${py_script} ${exp_id}


py_script="python pbmc_translate_to_cibersort.py -cs ${cybersort_path} -aug ${aug_data_path} -pidx 7 --no_use_test  -exp "
exp_id="pbmc3k"
lsf_file=${cybersort_path}/${exp_id}_7_translate_to_cibersort.lsf
bsub -R "rusage[mem=15GB]" -W 24:00 -n 1 -q "normal" -o ${lsf_file} -J ${exp_id} ${py_script} ${exp_id}


py_script="python pbmc_translate_to_cibersort.py -cs ${cybersort_path} -aug ${aug_data_path} -pidx 8 --no_use_test  -exp "
exp_id="pbmc3k"
lsf_file=${cybersort_path}/${exp_id}_8_translate_to_cibersort.lsf
bsub -R "rusage[mem=15GB]" -W 24:00 -n 1 -q "normal" -o ${lsf_file} -J ${exp_id} ${py_script} ${exp_id}


py_script="python pbmc_translate_to_cibersort.py -cs ${cybersort_path} -aug ${aug_data_path} -pidx 9 --no_use_test  -exp "
exp_id="pbmc3k"
lsf_file=${cybersort_path}/${exp_id}_9_translate_to_cibersort.lsf
bsub -R "rusage[mem=15GB]" -W 24:00 -n 1 -q "normal" -o ${lsf_file} -J ${exp_id} ${py_script} ${exp_id}


py_script="python pbmc_translate_to_cibersort.py -cs ${cybersort_path} -aug ${aug_data_path} -pidx 6 --use_test  -exp "
exp_id="pbmc3k"
lsf_file=${cybersort_path}/${exp_id}_6_translate_to_cibersort.lsf
bsub -R "rusage[mem=15GB]" -W 24:00 -n 1 -q "normal" -o ${lsf_file} -J ${exp_id} ${py_script} ${exp_id}



py_script="python pbmc_translate_to_cibersort.py -cs ${cybersort_path} -aug ${aug_data_path} -pidx 6 --no_use_test  -exp "
exp_id="pbmc3k"
lsf_file=${cybersort_path}/${exp_id}_6_translate_to_cibersort.lsf
bsub -R "rusage[mem=15GB]" -W 24:00 -n 1 -q "normal" -o ${lsf_file} -J ${exp_id} ${py_script} ${exp_id}

