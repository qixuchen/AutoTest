
# RT_Bench & ST_Bench: two manually labeled 1200-column benchmarks

Each row in `RT-bench` and `ST-bench` corresponds to a table column. 

Each row is described by the following fields:

`header`: header of the column

`ground_truth`: Set of **obvious** errors in this column (if any) 

`ground_truth_debateable`: Set of **contingent** errors in this column (if any)

`dist_val` : a list of distinct column values

`dist_val_count` : number of distinct column values

