methods="our-rule-list rlnet mdl-rule-list ripper xgboost"
for method in $methods
do
    echo "$method"
    echo "# Date: $(date)" > "results/real-world/multi-class-$method.csv"
    echo "Dataset;n_rules;F1;F1_std;Acc;Acc_std;AUC;AUC_std;Runtime;Runtime_std" >> results/real-world/multi-class-$method.csv
    for dataset in  satimage penguins iris car ecoli yeast
    do
        python run_method.py --method $method --dataset $dataset --n_rules 10 --outpath "results/real-world/multi-class-" 
    done
done