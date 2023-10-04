import os
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from sklearn.metrics import roc_auc_score
from torch.utils.tensorboard import SummaryWriter
import time
from datetime import timedelta

N_CLASSES = 14
CLASS_NAMES = [ 'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
                'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']
BATCH_SIZE = 16
IMAGES_PATH = 'data_chexnet/'
# Paths to the files with training, validation and testing sets.
# Each file should contains pairs [path to image, output vector]
# Example: images_011/00027736_001.png 0 0 0 0 0 0 0 0 0 0 0 0 0 0
SPLIT_PATH = 'split/'
MODELS_PATH = os.path.join(os.getcwd(), "models") # trained models
if not os.path.isdir(MODELS_PATH):
    os.mkdir(MODELS_PATH)
TB_NAME = 'baseline'
CKPT_PATH = os.path.join(MODELS_PATH, f'best_model_{TB_NAME}.pth')

def main():

    if torch.cuda.is_available():
        ngpus_per_node = torch.cuda.device_count()
        for ngpu in range(ngpus_per_node):
            print(f'Device {ngpu}', torch.cuda.get_device_name(ngpu))
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f'Running on {device}')

    cudnn.benchmark = True

    normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])
    transf_train = transforms.Compose([
        transforms.Resize(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])
    transf_val = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        normalize
    ])
    transf_test = transforms.Compose([
        transforms.Resize(256),
        transforms.TenCrop(224),
        transforms.Lambda
        (lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
        transforms.Lambda
        (lambda crops: torch.stack([normalize(crop) for crop in crops]))
        ])

    train_dataset = DatasetTrain(path_data=os.path.join(SPLIT_PATH, 'train_1.txt'), path_image=IMAGES_PATH, transf=transf_train)
    val_dataset = DatasetTrain(path_data=os.path.join(SPLIT_PATH, 'val_1.txt'), path_image=IMAGES_PATH, transf=transf_val)
    test_dataset = DatasetTrain(path_data=os.path.join(SPLIT_PATH, 'test_1.txt'), path_image=IMAGES_PATH, transf=transf_test)
    
    train(train_dataset, val_dataset, max_epochs=100, device=device)
    test(test_dataset, device=device)


def train(train_ds, val_ds, max_epochs, device):
    since = time.time()
    writer = SummaryWriter(log_dir='logs/train/'+TB_NAME)

    train_loader = torch.utils.data.DataLoader(dataset=train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(dataset=val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=True)

    model = CheXNet(N_CLASSES).to(device)
    model = torch.nn.DataParallel(model).to(device)
    loss_function = torch.nn.BCELoss(reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=5, mode='min')

    print("Training")
    best_loss = float('inf')
    step_len = len(train_ds) // train_loader.batch_size
    for epoch in range(max_epochs):
        print(f"epoch {epoch + 1}/{max_epochs}")
        model.train()
        epoch_loss_train = 0
        for step, batch_data in enumerate(train_loader):
            inputs, labels = batch_data[0].to(device, non_blocking=True), batch_data[1].to(device, non_blocking=True)
            optimizer.zero_grad()
            output = model(inputs)
            loss = loss_function(output, labels)
            loss.backward()
            optimizer.step()
            print(f"{step:3d}/{step_len}, train_loss: {loss.item():.4f}", end='\r')
            epoch_loss_train += loss.item()
        epoch_loss_train /= step_len
        writer.add_scalars('Loss', {'train': epoch_loss_train}, epoch)

        print('')
        print("Validation")
        model.eval()
        epoch_loss_val = 0
        loss_tensor = 0
        with torch.no_grad():
            for step, batch_data in enumerate(val_loader):
                inputs, labels = batch_data[0].to(device, non_blocking=True), batch_data[1].to(device, non_blocking=True)
                output = model(inputs)
                loss = loss_function(output, labels)
                loss_tensor += loss
                epoch_loss_val += loss.item()
            epoch_loss_val /= step + 1
            loss_tensor /= step + 1

        if epoch_loss_val < best_loss:
            best_loss = epoch_loss_val
            torch.save(model.state_dict(), CKPT_PATH)
            print("saved new best loss model")
        
        scheduler.step(loss_tensor.item())
        writer.add_scalars('Loss', {'validation': epoch_loss_val}, epoch)
        torch.cuda.empty_cache()
    
    time_elapsed = time.time() - since
    print(f'Time: {str(timedelta(seconds=time_elapsed))}')
    writer.add_text('time', f'Training complete in {str(timedelta(seconds=time_elapsed))}')
    writer.close()

def test(test_ds, device):
    writer = SummaryWriter(log_dir='logs/test/'+TB_NAME)

    print("Testing the existing model")
    model = CheXNet(N_CLASSES).to(device)
    model = torch.nn.DataParallel(model).to(device)
    if os.path.isfile(CKPT_PATH):
        print("=> loading checkpoint")
        model.load_state_dict(torch.load(CKPT_PATH))
        print("=> loaded checkpoint")
    else:
        print("=> no checkpoint found")
        return None

    test_loader = torch.utils.data.DataLoader(dataset=test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=True)
    
    # initialize the ground truth and output tensor
    gt = torch.tensor([], dtype=torch.float32, device=device)
    pred = torch.tensor([], dtype=torch.float32, device=device)

    model.eval()
    step_len = len(test_ds) // test_loader.batch_size
    for i, (inp, target) in enumerate(test_loader):
        target = target.to(device, non_blocking=True)
        gt = torch.cat((gt, target), 0)
        bs, n_crops, c, h, w = inp.size()
        with torch.no_grad():
            input_var = inp.view(-1, c, h, w).to(device, non_blocking=True)
            output = model(input_var)
            output_mean = output.view(bs, n_crops, -1).mean(1) # averaging predictions from different crops. result: 14 classes, each one with the probability of yes or no
        pred = torch.cat((pred, output_mean.data), 0) # concatenates the model's output (probabilities)
        print(f"{i:3d}/{step_len}", end='\r')

    gt = gt.cpu().numpy()
    pred = pred.cpu().numpy()

    AUROCs = compute_AUCs(gt, pred)
    valid_AUROCs = [a for a in AUROCs if a is not None]
    AUROC_avg = np.array(valid_AUROCs).mean()
    print(f'The average AUROC is {AUROC_avg:.3f}')
    writer.add_text('Average AUROC', str(AUROC_avg))
    for i in range(N_CLASSES):
        print(f'The AUROC of {CLASS_NAMES[i]} is {AUROCs[i]}')
    
    # Log AUROCs in TensorBoard
    if writer is not None:
        table = f"""
            | {'Class Name':<18} | AUROC |
            |{'-' * 20}|-------|
            """
        for i in range(N_CLASSES):
            table += f"""
            | {CLASS_NAMES[i]:<18} | {AUROCs[i]:.4f} |
            """

        writer.add_text('AUROCs_Table', table)
    
    writer.close()

def compute_AUCs(gt_np, pred_probs_np):
    """Computes Area Under the Curve (AUC) from prediction scores.

    Args:
        gt: Pytorch tensor on GPU, shape = [n_samples, n_classes]
          true labels.
        pred_probs: Pytorch tensor on GPU, shape = [n_samples, n_classes]
          probability estimates of the positive class

    Returns:
        List of AUROCs of all classes.
    """
    AUROCs = []
    for i in range(N_CLASSES):
        try:
            AUROCs.append(roc_auc_score(gt_np[:, i], pred_probs_np[:, i]))
        except ValueError as e:
            if "Only one class present in y_true" in str(e):
                print(f"Skipping AUROC calculation for class {CLASS_NAMES[i]}: {str(e)}")
                AUROCs.append(None)  # Add None for this class to indicate the issue
            else:
                raise  # Re-raise the exception if it's not the expected one
    return AUROCs


class CheXNet(nn.Module):
    """Model modified.

    The architecture of our model is the same as standard DenseNet121
    except the classifier layer which has an additional sigmoid function.

    """
    def __init__(self, out_size):
        super(CheXNet, self).__init__()
        self.densenet121 = torchvision.models.densenet121(weights='DEFAULT')
        num_ftrs = self.densenet121.classifier.in_features
        self.densenet121.classifier = nn.Sequential(
            nn.Linear(num_ftrs, out_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.densenet121(x)
        return x
    
class DatasetTrain(torch.utils.data.Dataset):
    def __init__(self, path_data, path_image, transf=None):

        self.image_files = []
        self.labels = []
        self.transform = transf

        with open(path_data, "r") as file:
            for line in file:
                line_items = line.strip().split()
                if len(line_items) < 2:
                    continue

                image_path = os.path.join(path_image, line_items[0])
                image_label = list(map(int, line_items[1:]))

                self.image_files.append(image_path)
                self.labels.append(image_label)


    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        image = Image.open(self.image_files[index]).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image, torch.tensor(self.labels[index], dtype=torch.float32)

if __name__ == '__main__':
    main()