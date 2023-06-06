#! /bin/bash
set -x

python demo_folder.py -y ./config/semantickitti.yaml --demo-folder ./work/infer_test/velodyne --save-folder ./work/infer_test/labels/
