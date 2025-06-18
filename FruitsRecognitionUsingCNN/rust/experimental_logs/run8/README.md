stopped training after 32 %

done with 64 batch size, valid padding, kernel [2,2], normal rmspropconfig

Finished `dev` profile [unoptimized + debuginfo] target(s) in 0.27s
Running `target/debug/rust`
Loading the dataset needed 237.44870 ms
Balancing the dataset needed 146.78256 ms
Splitting test set into validation- and test set needed 3.05200 ms
Training set shape: (101140, 100, 100, 3)
Validation set shape: (7000, 100, 100, 3)
Test set shape: (26769, 100, 100, 3)
Data Preprocessing needed 675149.38901 ms
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
Training CNN and Saving Model needed 1630776.58762 ms
Prediction needed 25304.34722 ms
Logs saved to ./artifacts/cnn_time_log_20250604-114722.txt

Test accuracy: 0.0120
