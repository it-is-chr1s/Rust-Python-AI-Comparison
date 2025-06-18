changed CNN filters from valid padding to same padding and kernel size to [3, 3]

running on GPU 2

like in python
RmsPropConfig::new().with_alpha(0.9).with_epsilon(1e-7),

(base) student@srvgpu:~/CNN-FruitsRecognition/rust$ cargo run
Finished `dev` profile [unoptimized + debuginfo] target(s) in 0.26s
Running `target/debug/rust`
Loading the dataset needed 230.20327 ms
Balancing the dataset needed 146.52253 ms
Splitting test set into validation- and test set needed 2.97473 ms
Training set shape: (101140, 100, 100, 3)
Validation set shape: (7000, 100, 100, 3)
Test set shape: (26769, 100, 100, 3)
Data Preprocessing needed 673749.85267 ms
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
Training CNN and Saving Model needed 5231432.65216 ms
Prediction needed 26023.91539 ms
Logs saved to ./artifacts/cnn_time_log_20250604-000330.txt

Test accuracy: 0.1971
