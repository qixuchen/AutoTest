
# RT_Bench & ST_Bench: manually-labeled 1200-column benchmarks

In this folder, we have two 1200-column benchmarks `RT-bench` (randomly sampled from real relational-tables), and `ST-bench` (randomly sampled from real spreadsheet-tables).

Each row in `RT-bench` and `ST-bench` corresponds to a real table column, which may contain real errors that are manually inspected and labeled.

Each row in the benchmark files has by the following 5 fields:
- `header`: this is the original header of the column

- `ground_truth`: this field contains all data errors in the column, that are labeled as **unambiguous** and **obvious** errors (e.g., misspelled values like 'febuary'). An algorithm that misses  errors in `ground_truth` will be counted as a recall loss.

- `ground_truth_debateable`: this field contains data errors that are **debatable** in nature (e.g., in a column ['Q1', 'Q2', 'Q3', 'Q4', 'total'], it may be debatable whether the cell value "total" should be regarded as an error, as people may have differing opinions). Given the debatable nature of such values, predictions made or missed by an algorithm for values in `ground_truth_debateable` will not affect its precision or recall.

- `dist_val` : the full list of all distinct values in this column. An algorithm that predicts any values in this column, other than the values listed in  `ground_truth` and `ground_truth_debateable`, will count as precision loss.

- `dist_val_count` : the total number of distinct values in this column.

