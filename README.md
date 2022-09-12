# dcase2022
Submission for task 2 "Unsupervised Anomalous Sound Detection for Machine Condition Monitoring Applying Domain Generalization Techniques" of the DCASE challenge 2022 (https://dcase.community/challenge2022/task-unsupervised-anomalous-sound-detection-for-machine-condition-monitoring). The system is a conceptually simple outlier exposed ASD system specifically designed for domain generalization and uses the sub-cluster AdaCos loss (https://github.com/wilkinghoff/sub-cluster-AdaCos). A detailed description of the system can be found here: https://dcase.community/documents/challenge2022/technical_reports/DCASE2022_Wilkinghoff_2_t2.pdf

The implementation is based on Tensorflow 2.3 (more recent versions can run into problems with the current implementation). Just start the main script for training and evaluation. To run the code, you need to download the development dataset, additional training dataset and the evaluation dataset, and store the files in an './eval_data' and a './dev_data' folder.

When finding this code helpful, or reusing parts of it, a citation would be appreciated:
@techreport{wilkinghoff2022outlier,
  title={An outlier exposed anomalous sound detection system for domain generalization in machine condition monitoring},
  author={Wilkinghoff, Kevin},
  year={2022},
  institution={DCASE2022 Challenge, Tech. Rep}
}
