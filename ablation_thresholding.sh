# Create the output CSV and append header info


cleanup() {
    echo "Terminating all running jobs..."
    kill $(jobs -p) 2>/dev/null
    exit 1
}

# Trap SIGINT (Ctrl+C) and call cleanup function
trap 'cleanup' SIGINT
# Function to run the Python script
run_script() {
    python run_method.py --method our-rule-list --dataset "$1" --n_rules $2 --outpath "results/ablation/ablation-thresholding-kmeans-" --ablation_thresholding 1 --ablation_thresholding_kind kmeans &
}

for n_rules in 10
    do 

    # Maximum number of parallel jobs
    max_parallel=10

    # Function to wait if maximum parallel jobs are running
    wait_for_jobs() {
        while [[ $(jobs -r | wc -l) -ge $max_parallel ]]; do
            sleep 1  # Wait for 1 second before checking again
        done
    }

    echo "# Date: $(date)" > "results/ablation/ablation-thresholding-kmeans-our-rule-list.csv"
    echo "Dataset;n_rules;F1;F1_std;Acc;Acc_std;AUC;AUC_std;Runtime;Runtime_std" >> "results/ablation/ablation-thresholding-kmeans-our-rule-list.csv"

    # First batch of datasets
    dsets="heart credit_g covid phoneme qsar_biodeg diabetes hepatitis titanic tokyo1 crx ring electricity phishing android juvenile_clean magic compas_two_year_clean fico credit_card_clean adult"
    for dataset in $dsets
    do
        run_script "$dataset" $n_rules
        wait_for_jobs
    done
done
wait # Wait for all remaining jobs

run_script2() {
    python run_method.py --method our-rule-list --dataset "$1" --n_rules $2 --outpath "results/ablation/ablation-thresholding-uniform-" --ablation_thresholding 1 --ablation_thresholding_kind uniform &
}

for n_rules in 10 # 15 20
    do 

    # Maximum number of parallel jobs
    max_parallel=10

    # Function to wait if maximum parallel jobs are running
    wait_for_jobs() {
        while [[ $(jobs -r | wc -l) -ge $max_parallel ]]; do
            sleep 1  # Wait for 1 second before checking again
        done
    }

    echo "# Date: $(date)" > "results/ablation/ablation-thresholding-uniform-our-rule-list.csv"
    echo "# Commit-Hash: $(git rev-parse HEAD)" >> "results/ablation/ablation-thresholding-uniform-our-rule-list.csv"
    echo "Dataset;n_rules;F1;F1_std;Acc;Acc_std;AUC;AUC_std;Runtime;Runtime_std" >> "results/ablation/ablation-thresholding-uniform-our-rule-list.csv"

    # First batch of datasets
    dsets="heart credit_g covid phoneme qsar_biodeg diabetes hepatitis titanic tokyo1 crx ring electricity phishing android juvenile_clean magic compas_two_year_clean fico credit_card_clean adult"
    for dataset in $dsets
    do
        run_script2 "$dataset" $n_rules
        wait_for_jobs
    done
done
wait
exit 0
