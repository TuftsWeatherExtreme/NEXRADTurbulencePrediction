# train_and_test_model.py
# Team Celestial Blue
# Spring 2025
# Last Modified: 05/06/2025
# Description: This script trains and evaluates a model using K-Fold 
#   cross-validation.
#   Features: 
#    - Supports both linear and hybrid models, and allows for different 
#   loss functions.
#    - Saves checkpoints during training to allow for seamless resumption
#       if interrupted while training
#    - Determines the best model and evaluates it on a held-out test set
#    - Saves the best model to 
#       trained_model_outputs/{timestamp}_best_{model_type}_mse_model_w_seed_{SEED}.pth

import os
import torch 
import torch.nn as nn
import sys

# output to timestamped file
DIRNAME = os.path.dirname(sys.argv[0])

# Append to sys path to import scale_turbulence from plane_weights
sys.path.append(os.path.join(DIRNAME, "..",))

from model_architecture.regression_model import LinearClassifierModel
from model_architecture.hybrid_model_1_out import HybridModel1Out
from model_architecture.hybrid_model import HybridModel
from model_architecture.resnet3d_model import ResNet3DModel
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import time, datetime
from dataloader_class import RadarDataLoader
from sklearn.model_selection import KFold
from torch.utils.data import Subset
import torch.nn.functional as F

NUM_EPOCHS = 30
BATCH_SIZE = 64
NUM_FOLDS = 6
# TODO: Set DATALOADER_PATH to dataloader we want to use for training
# DATALOADER_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eleanor_dataloader.pth")
DATALOADER_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nehir_dataloader.pth")

terminate_training = False
loss_is_nll = False

def usage():
    print("Usage: python train_and_test_model.py [linear|hybrid|resnet] [SEED]")
    print("linear: Train a linear classifier")
    print("hybrid: Train the hybrid CNN + FC model")
    print("resnet: Train the ResNet3D model")
    print("SEED: seed for splitting dataset and saving model checkpoint")
    exit(1)

if len(sys.argv) != 3:
    usage()
try:
    SEED = int(sys.argv[2])
except:
    usage()

MODEL_TYPE = sys.argv[1]

OUTPUT_DIR = os.path.join(DIRNAME, "trained_model_outputs")
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

curr_time = time.time()
formatted_curr_date = datetime.datetime.fromtimestamp(curr_time).isoformat()

MODEL_CHECKPOINT_PATH = os.path.join(OUTPUT_DIR, f"{MODEL_TYPE}_{SEED}_model_checkpoint.pth")
OUTPUT_FILENAME = os.path.join(OUTPUT_DIR, formatted_curr_date + f"_best_{MODEL_TYPE}_model_w_seed_{SEED}")
RESULTS_FILEPATH = OUTPUT_FILENAME + "_results.txt"
RESULTS_FILE = open(RESULTS_FILEPATH, "a")
MODEL_FILEPATH = OUTPUT_FILENAME + ".pth"

device = torch.device("cpu")
if torch.cuda.is_available(): 
    device = torch.device("cuda")
    print("Using GPU!!!")

def save_checkpoint(model, optimizer, l2_alpha_idx, fold, epoch, l2_loss_list, loss_per_fold_list, loss_per_epoch_list):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'fold': fold,
        'l2_alpha_idx': l2_alpha_idx,
        'l2_loss_list': l2_loss_list,
        'loss_per_fold_list': loss_per_fold_list,
        'loss_per_epoch_list': loss_per_epoch_list
    }
    torch.save(checkpoint, MODEL_CHECKPOINT_PATH)
    print(f"Checkpoint saved at {MODEL_CHECKPOINT_PATH} for l2_alpha_idx: {l2_alpha_idx}, fold {fold}, and epoch {epoch}")


def load_checkpoint(model, optimizer):
    if os.path.exists(MODEL_CHECKPOINT_PATH):
        checkpoint = torch.load(MODEL_CHECKPOINT_PATH, weights_only=False)
        print(checkpoint['model_state_dict'])
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint
    else:
        print("No checkpoint found.")
        return None
        
        
def init_loaders(train_idx, val_idx, dataset, weights):
    train_subset = Subset(dataset, train_idx)
    # Create sampler for just the training subset
    subset_weights = [weights[i] for i in train_idx]
    subset_sampler = torch.utils.data.WeightedRandomSampler(subset_weights, num_samples=len(subset_weights), replacement=True)
    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, sampler=subset_sampler)

    val_subset = Subset(dataset, val_idx)
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=True)
    print(f"Train subset size: {len(train_subset)}, Validation subset size: {len(val_subset)}")
    return train_loader, val_loader


def train_model(model, epoch, train_loader, optimizer, loss_fn, verbose=False):
    running_train_loss = 0.0
    for batch_num, (x_train, y_train) in enumerate(train_loader):
        x_train = x_train.to(device)
        y_train = y_train.long().to(device)

        optimizer.zero_grad()
        y_hat = model(x_train)
        loss = loss_fn(y_hat, y_train)

        loss.backward()
        optimizer.step()
        running_train_loss += loss.item()

        if verbose and batch_num % 50 == 49:
            print(f'[Epoch {epoch}, Batch {batch_num + 1:5d}] avg loss: {running_train_loss / 50:.4f}')
            running_train_loss = 0.0

def evaluate_model(model, val_loader, loss_fn, verbose=False):
    running_valid_loss = 0.0
    with torch.no_grad():
        for batch_num, (x_val, y_val) in enumerate(val_loader):
            x_val = x_val.to(device)
            y_val = y_val.long().to(device)

            y_hat_val = model(x_val)
            val_loss = loss_fn(y_hat_val, y_val)
            running_valid_loss += val_loss.item()

    avg_valid_loss_epoch = running_valid_loss / len(val_loader)
    print(f"Validation loss: {avg_valid_loss_epoch:.4f} ({len(val_loader)} batches)")
    return avg_valid_loss_epoch

def train_and_eval_epoch(model, epoch, train_loader, val_loader, optimizer, loss_fn, fold, save_model=False):
    # ----------------- TRAINING ----------------- 
    print(f"BEGINNING EPOCH {epoch}")
    model.train()
    train_model(model, epoch, train_loader, optimizer, loss_fn, verbose=True)

    # ----------------- VALIDATION ----------------- 
    model.eval()
    avg_valid_loss_epoch = evaluate_model(model, val_loader, loss_fn, verbose=True)

    return avg_valid_loss_epoch




def main():
    model_type = sys.argv[1]
    MODELS = {
        "linear": LinearClassifierModel,
        "hybrid": HybridModel1Out,
        "resnet": ResNet3DModel,
    }
    if model_type not in MODELS:
        print(f"Unknown model type: {model_type}. Choose from: {', '.join(MODELS.keys())}")
        sys.exit(1)
    Model = MODELS[model_type]
    
    # load in pickled dataset from file and instantiate DataLoader Object
    dataset = torch.load(DATALOADER_PATH, weights_only=False) # load in saved dataset
    
    # WORKING ON THIS
    print("Original dataset length:", len(dataset))

    # Split dataset into 85% train, 15% test
    dataset, test_dataset = torch.utils.data.random_split(dataset, [0.85, 0.15], generator=torch.Generator().manual_seed(SEED))
    
    
    # NEW CODE, MIGHT NEED TO DEBUG!!
    labels = [label for _, label in dataset]
    num_nonsevere = labels.count(0.0)
    num_severe = labels.count(1.0)
    pos_weight = torch.tensor([num_nonsevere / num_severe]).to(device)
    weights = [1.0 / num_nonsevere if l == 0.0 else 1.0 / num_severe for l in labels]
    sampler = torch.utils.data.WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
    all_train_dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=sampler)
    
    
    

    # Create DataLoaders
    #all_train_dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # https://medium.com/biased-algorithms/cross-validation-in-pytorch-2f9f9fa9ab16
    # Initialize KFold
    kfold = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)

    # create a list of L2 penalties to test
    l2_alpha_list = [0.10, 0.01, 0.001, 0]
    # to store the average loss (over all folds) with each regularization parameter
    l2_loss_list = list()

    # Cross-entropy loss for 2-class classification
    loss_fn = nn.CrossEntropyLoss()

    print(f"Training the {model_type} model")
    print(f"Using CrossEntropyLoss")
    print(f"Using {SEED} as the seed for splitting the dataset")

    model = Model().to(device)
    optimizer = optim.AdamW(params=model.parameters(), lr=1e-3)

    restarting_from_checkpoint = False
    checkpoint = load_checkpoint(model, optimizer)
    if checkpoint is None:
        start_l2_alpha_idx = 0
        start_fold = 0
        start_epoch = 0
    else:
        restarting_from_checkpoint = True
        start_l2_alpha_idx = checkpoint['l2_alpha_idx']
        start_fold = checkpoint['fold']
        start_epoch = checkpoint['epoch']
        l2_loss_list = checkpoint['l2_loss_list']
        loss_per_fold_list = checkpoint['loss_per_fold_list']
        loss_per_epoch_list = checkpoint['loss_per_epoch_list']
        print(f"Checkpoint loaded from {MODEL_CHECKPOINT_PATH} for l2_alpha_idx: {start_l2_alpha_idx}, fold {start_fold}, and epoch {start_epoch}")
        print(f"l2_loss_list: {l2_loss_list}, loss_per_fold_list: {loss_per_fold_list}, loss_per_epoch_list: {loss_per_epoch_list}")

    print(f"Beginning search over different values of l2_alpha\n")
    for l2_alpha_idx in range(start_l2_alpha_idx, len(l2_alpha_list)):
        # To store the loss for each fold
        l2_alpha = l2_alpha_list[l2_alpha_idx]
        loss_per_fold_list = loss_per_fold_list if restarting_from_checkpoint else list()
        
        print(f"Beginning {NUM_FOLDS}-fold Cross Validation with l2_alpha = {l2_alpha}")
        # Split the dataset into 6 folds
        

        for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
            if fold < start_fold: # Skip folds that have already been trained
                continue
            print(f"Fold {fold} with l2_alpha = {l2_alpha}")
            loss_per_epoch_list = loss_per_epoch_list if restarting_from_checkpoint else list()
            train_loader, val_loader = init_loaders(train_idx, val_idx, dataset, weights)
            model = model if restarting_from_checkpoint else Model().to(device)
            optimizer = optimizer if restarting_from_checkpoint else optim.AdamW(model.parameters(), lr=1e-3, weight_decay=l2_alpha)

            for epoch in range(start_epoch + 1, NUM_EPOCHS):
                restarting_from_checkpoint = False
                avg_val_loss = train_and_eval_epoch(model, epoch, train_loader, val_loader, optimizer, loss_fn, fold, save_model=True)
                loss_per_epoch_list.append(avg_val_loss)
                print(f"loss_per_epoch_list on epoch {epoch} is {loss_per_epoch_list}\n")
                save_checkpoint(model, optimizer, l2_alpha_idx, fold, epoch, l2_loss_list, loss_per_fold_list, loss_per_epoch_list)

            # Save results for this fold (always last epoch)
            loss_per_fold_list.append(loss_per_epoch_list[-1])
            print(f"Fold {fold} had loss: {loss_per_fold_list[-1]} on its last epoch\n")  

        # Calculate the average loss for this fold
        fold_avg_loss = sum(loss_per_fold_list) / len(loss_per_fold_list)
        l2_loss_list.append(fold_avg_loss)
        print(f"l2_loss_list is now {l2_loss_list}\n")

    print(f"length of l2_loss_list is {len(l2_loss_list)}\n")
    print(f"l2_loss_list is {l2_loss_list}\n")
    best_l2_idx = np.argmin(l2_loss_list)
    best_l2_val = l2_alpha_list[best_l2_idx]
    print(f"best l2_alpha value is {best_l2_val} with a loss of: {np.min(l2_loss_list)}")

    print(f"Completed {NUM_FOLDS}-fold Cross Validation")

    best_model = Model().to(device)
    optimizer = optim.AdamW(best_model.parameters(), lr=1e-3, weight_decay=best_l2_val)

    # Retrain on the 90 percent of the data
    print("Retraining best model on 90 percent of the data\n")
    for epoch in range(NUM_EPOCHS):
        for batch_num, data in enumerate(all_train_dataloader):
            x_train, y_train = data

            x_train, y_train = x_train.to(device), y_train.long().to(device)

            optimizer.zero_grad() # zero the parameter gradients at the beginning
            y_hat = best_model(x_train)

            loss = loss_fn(input=y_hat, target=y_train)
                    
            loss.backward()
            optimizer.step()

    print("Testing the retrained best model\n")

    # ----------------- TESTING -----------------
    num_correct = 0
    num_false_positive = 0
    num_false_negative = 0
    num_true_positive = 0
    total = 0
    
    best_model.eval()
    with torch.no_grad():
        for x_test, y_test in test_dataloader:
            x_test, y_test = x_test.to(device), y_test.long().to(device)
            y_hat_test = best_model(x_test)
            predicted = torch.argmax(y_hat_test, dim=1).float()
            for i in range(len(y_test)):
                true_label = int(y_test[i].item())
                pred_label = int(predicted[i].item())
                total += 1
                if true_label == pred_label:
                    num_correct += 1
                if true_label == 0 and pred_label == 1:
                    num_false_positive += 1
                if true_label == 1 and pred_label == 0:
                    num_false_negative += 1
                if true_label == 1 and pred_label == 1:
                    num_true_positive += 1
    
    precision = num_true_positive / (num_true_positive + num_false_positive + 1e-8)
    recall = num_true_positive / (num_true_positive + num_false_negative + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    
    results = [
        f"Accuracy: {num_correct}/{total} = {num_correct/total*100:.2f}%",
        f"False positive rate: {num_false_positive}/{total} = {num_false_positive/total*100:.2f}%",
        f"False negative rate: {num_false_negative}/{total} = {num_false_negative/total*100:.2f}%",
        f"Precision: {precision:.4f}",
        f"Recall: {recall:.4f}",
        f"F1 Score: {f1:.4f}"
    ]
    for r in results:
        print(r)
        RESULTS_FILE.write(r + "\n")

    # Save the best model
    torch.save(best_model.state_dict(), MODEL_FILEPATH)
    # Remove the checkpoint file
    os.remove(MODEL_CHECKPOINT_PATH)
    RESULTS_FILE.close()

if __name__ == "__main__":
    main()
