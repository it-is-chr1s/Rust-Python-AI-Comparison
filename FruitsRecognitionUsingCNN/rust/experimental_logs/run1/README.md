used wgpu as backend
and a batch size of 8 because the wgpu wasn't able to handle a larger batch size

Finished `dev` profile [unoptimized + debuginfo] target(s) in 0.27s
Running `target/debug/rust`
Loading the dataset needed 229.21634 ms
Balancing the dataset needed 146.70941 ms
Splitting test set into validation- and test set needed 3.77652 ms
Training set shape: (101140, 100, 100, 3)
Validation set shape: (7000, 100, 100, 3)
Test set shape: (26769, 100, 100, 3)
Data Preprocessing needed 679411.40234 ms
Dataset is ready for batching.
Shape dataset_train: (Images: (101140, 3, 100, 100), Labels: (101140))
Shape dataset_valid: (Images: (7000, 3, 100, 100), Labels: (7000))
Create artifact dir
Configure seed
Initialise batcher
Initialise dataloader_train
Initialise dataloader_valid
Initialise learner
Batch size: 8
Train dataset size: 101140
Valid dataset size: 7000
Devices: [DefaultDevice]
Number of devices: 1
Device type: cubecl_wgpu::device::WgpuDevice
Training CNN and Saving Model needed 23361170.91163 ms
Prediction needed 66291.06440 ms

Test accuracy: 0.0123
