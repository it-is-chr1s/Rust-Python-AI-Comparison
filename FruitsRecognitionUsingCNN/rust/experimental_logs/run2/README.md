used tch backend
and a batchsize of 32 because it could handle it finally

(base) student@srvgpu:~/CNN-FruitsRecognition/rust$ cargo run
Finished `dev` profile [unoptimized + debuginfo] target(s) in 0.27s
Running `target/debug/rust`
Loading the dataset needed 247.14857 ms
Balancing the dataset needed 146.01216 ms
Splitting test set into validation- and test set needed 2.91563 ms
Training set shape: (101140, 100, 100, 3)
Validation set shape: (7000, 100, 100, 3)
Test set shape: (26769, 100, 100, 3)
Data Preprocessing needed 669388.86608 ms
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
Devices: [Cuda(0)]
Number of devices: 1
Device type: burn_tch::backend::LibTorchDevice
Training CNN and Saving Model needed 3810800.61964 ms
Prediction needed 25720.76909 ms
Logs saved to ./artifacts/cnn_time_log_20250603-184222.txt

Test accuracy: 0.4826
