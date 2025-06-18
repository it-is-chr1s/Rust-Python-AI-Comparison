import os
import matplotlib.pyplot as plt
from datetime import datetime

def read_metric_logs(base_dir):
    accs = []
    losses = []

    epoch_dirs = sorted(
        [d for d in os.listdir(base_dir) if d.startswith("epoch-")],
        key=lambda x: int(x.split("-")[1])
    )

    for epoch_dir in epoch_dirs:
        acc_path = os.path.join(base_dir, epoch_dir, "Accuracy.log")
        loss_path = os.path.join(base_dir, epoch_dir, "Loss.log")

        with open(acc_path, "r") as f:
            acc = float(f.read().strip())
        with open(loss_path, "r") as f:
            loss = float(f.read().strip())

        accs.append(acc)
        losses.append(loss)

    return accs, losses

train_acc, train_loss = read_metric_logs('./artifacts/train')
valid_acc, valid_loss = read_metric_logs('./artifacts/valid')

# Plotting
plt.figure(1)
plt.subplot(211)
plt.plot(train_acc)
plt.plot(valid_acc)
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Valid'], loc='upper left')

plt.subplot(212)
plt.plot(train_loss)
plt.plot(valid_loss)
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Valid'], loc='upper left')

timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
plt.savefig(f"./artifacts/visualizations/training_visualization_{timestamp}.png")
plt.close()