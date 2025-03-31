#! /bin/sh

# For Rt_train
python3 STEP1_SDC_generation.py rt_train
python3 STEP2_SDC_selection.py rt_train
python3 STEP3_SDC_application.py rt_bench rt_train
python3 ./AutoTest_reproduce_main_results.py rt_bench rt_train
python3 STEP3_SDC_application.py st_bench rt_train
python3 ./AutoTest_reproduce_main_results.py st_bench rt_train

# For Tablib
python3 STEP1_SDC_generation.py tablib
python3 STEP2_SDC_selection.py tablib
python3 STEP3_SDC_application.py rt_bench tablib
python3 ./AutoTest_reproduce_main_results.py rt_bench tablib
python3 STEP3_SDC_application.py st_bench tablib
python3 ./AutoTest_reproduce_main_results.py st_bench tablib