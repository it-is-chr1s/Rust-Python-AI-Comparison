stopped by 43 % of training

done with same padding, kernel [3,3], and special rmsprop config

Finished `dev` profile [unoptimized + debuginfo] target(s) in 0.27s
Running `target/debug/rust`
Loading the dataset needed 247.72457 ms
Balancing the dataset needed 146.00185 ms
Splitting test set into validation- and test set needed 2.91975 ms
Training set shape: (101140, 100, 100, 3)
Validation set shape: (7000, 100, 100, 3)
Test set shape: (26769, 100, 100, 3)
Data Preprocessing needed 683801.99877 ms
Dataset is ready for batching.
Shape dataset_train: (Images: (101140, 3, 100, 100), Labels: (101140))
Shape dataset_valid: (Images: (7000, 3, 100, 100), Labels: (7000))
Create artifact dir
Configure seed
Initialise batcher
Initialise dataloader_train
Initialise dataloader_valid
Initialise learner
Batch size: 64
Train dataset size: 101140
Valid dataset size: 7000
Devices: [Cuda(2)]
Number of devices: 1
Device type: burn_tch::backend::LibTorchDevice
Training CNN and Saving Model needed 2293845.10520 ms
Prediction needed 25146.40722 ms
Logs saved to ./artifacts/cnn_time_log_20250604-104659.txt

Test accuracy: 0.5020
