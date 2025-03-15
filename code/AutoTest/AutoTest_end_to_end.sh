#! /bin/sh

python3 STEP1_SDC_generation.py tablib_small

python3 STEP2_SDC_selection.py tablib_small

python3 STEP3_SDC_application.py rt_bench tablib_small

python3 ./AutoTest_reproduce_main_results.py rt_bench tablib_small