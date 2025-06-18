running on GPU 2

like in python
RmsPropConfig::new().with_alpha(0.9).with_epsilon(1e-7),

Finished `dev` profile [unoptimized + debuginfo] target(s) in 0.26s
Running `target/debug/rust`
Loading the dataset needed 226.21656 ms
Balancing the dataset needed 147.63910 ms
Splitting test set into validation- and test set needed 2.97830 ms
Training set shape: (101140, 100, 100, 3)
Validation set shape: (7000, 100, 100, 3)
Test set shape: (26769, 100, 100, 3)
Data Preprocessing needed 678840.17630 ms
Dataset is ready for batching.
Shape dataset_train: (Images: (101140, 3, 100, 100), Labels: (101140))
Shape dataset_valid: (Images: (7000, 3, 100, 100), Labels: (7000))
Create artifact dir
Configure seed
Initialise batcher
Initialise dataloader_train
Initialise dataloader_valid
Initialise learner
Batch size: 32
Train dataset size: 101140
Valid dataset size: 7000
Devices: [Cuda(2)]
Number of devices: 1
Device type: burn_tch::backend::LibTorchDevice
Training CNN and Saving Model needed 3823575.10223 ms
Prediction needed 24608.85721 ms
Logs saved to ./artifacts/cnn_time_log_20250603-203737.txt

Test accuracy: 0.3367
