tried with same padding and kernel size [3,3]

no success, loss isn't decreasing, stopped after 5 epochs

(base) student@srvgpu:~/CNN-FruitsRecognition/rust$ cargo run
Compiling torch-sys v0.19.0
Compiling tch v0.19.0
Compiling burn-tch v0.17.0
Compiling burn v0.17.0
Compiling rust v0.1.0 (/data/home/student/CNN-FruitsRecognition/rust)
Finished `dev` profile [unoptimized + debuginfo] target(s) in 38.53s
Running `target/debug/rust`
Loading the dataset needed 264.72754 ms
Balancing the dataset needed 144.78467 ms
Splitting test set into validation- and test set needed 2.90268 ms
Training set shape: (101140, 100, 100, 3)
Validation set shape: (7000, 100, 100, 3)
Test set shape: (26769, 100, 100, 3)
Data Preprocessing needed 695462.74348 ms
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
Training CNN and Saving Model needed 2974906.29012 ms
Prediction needed 25317.51344 ms
Logs saved to ./artifacts/cnn_time_log_20250604-093913.txt

Test accuracy: 0.0123
