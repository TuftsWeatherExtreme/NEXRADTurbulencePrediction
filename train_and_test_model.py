import torch 
import torch.nn as nn
from regression_model import LinearClassifierModel
from hybrid_model_1_out import HybridModel1Out
from hybrid_model import HybridModel
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import time, datetime
from dataloader_class import RadarDataLoader
from sklearn.model_selection import KFold
from torch.utils.data import Subset
import os
import torch.nn.functional as F
import sys

NUM_EPOCHS = 5 
BATCH_SIZE = 2000
NUM_FOLDS = 6

terminate_training = False
loss_is_nll = False


# output to timestamped file
output_dir = "trained_model_outputs"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

MODEL_CHECKPOINT_PATH = ""
curr_time = time.time()
formatted_curr_date = datetime.datetime.fromtimestamp(curr_time).isoformat()

output_file_path = os.path.join(output_dir, formatted_curr_date + "results.txt")
output_file = open(output_file_path, "a")

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


def init_loaders(train_idx, val_idx, dataset):
    print(f"First {NUM_FOLDS} train indices: {train_idx[:NUM_FOLDS]}")
    print(f"First {NUM_FOLDS} validation indices: {val_idx[:NUM_FOLDS]}")

    # Create subsets for training and validation for current fold
    # Adapt the subsets into dataloaders
    train_subset = Subset(dataset, train_idx)
    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)

    val_subset = Subset(dataset, val_idx)
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=True)
    print(f"Train subset size: {len(train_subset)}, Validation subset size: {len(val_subset)}")

    return train_loader, val_loader


def train_model(model, epoch, train_loader, optimizer, loss_fn, verbose=False):
    # iterate through all the data
    running_train_loss = 0.0
    for batch_num, (x_train, y_train) in enumerate(train_loader):

        # Move data to the appropriate device
        x_train, y_train = x_train.to(device), y_train.float().to(device)

        optimizer.zero_grad() # zero the parameter gradients at the beginning
        y_hat = model(x_train) # Evaluate the model on the input data

        if loss_is_nll:
            y_hat = log_softmax(y_hat)
            y_train = y_train.long()

        loss = loss_fn(y_hat, y_train)

        loss.backward()
        optimizer.step()
        running_train_loss += loss.item() # Yields the average loss per batch

        if verbose and batch_num % 100 == 100 - 1:    # print every 100 mini-batches
            output_str = f'[Epoch: {epoch}, Batch_num: {batch_num + 1:5d}] avg training loss per batch: {running_train_loss / 100:.3f}'
            output_file.write(output_str)
            print(output_str)
            print(f"On batch: {batch_num + 1}, train_loss: {loss.item()}, running train loss: {running_train_loss}")
            running_train_loss = 0.0

def evaluate_model(model, val_loader, loss_fn, verbose=False):
    running_valid_loss = 0.0
    with torch.no_grad(): # disable gradient tracking for validation
        for batch_num, (x_val, y_val) in enumerate(val_loader):
            
            x_val, y_val = x_val.to(device), y_val.float().to(device)

            y_hat_val = model(x_val)

            if loss_is_nll:
                y_hat_val = log_softmax(y_hat_val)
                y_val = y_val.long()

            val_loss = loss_fn(y_hat_val, y_val)

            running_valid_loss += val_loss.item()
            if verbose and batch_num % 20 == 20 - 1:    # print every 20 mini-batches
                print(f"On batch: {batch_num + 1:3d}, val_loss: {val_loss.item()}, running val loss: {running_valid_loss}")
    print(f"Evaluated model on {len(val_loader)} batches of ~2000")
    avg_valid_loss_epoch = running_valid_loss / len(val_loader)
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

def usage():
    print("Usage: python train_and_test_model.py [linear|hybrid] [LOSS_TYPE] [SEED]")
    print("linear: Train a linear classifier")
    print("hybrid: Train a hybrid classifier")
    print("LOSS_TYPE: loss function to use (e.g., mse, mae, nll)")
    print("SEED: seed for splitting dataset and saving model checkpoint")
    exit(1)


def main():
    if len(sys.argv) != 4 or (sys.argv[1] != "linear" and sys.argv[1] != "hybrid"):
        usage()
    try:
        LOSS_TYPE = sys.argv[2]
        SEED = int(sys.argv[3])
    except:
        usage()
        raise f"Could not cast {sys.argv[3]} to an int"

    Model = LinearClassifierModel
    global MODEL_CHECKPOINT_PATH
    MODEL_CHECKPOINT_PATH = os.path.join(output_dir, f"{sys.argv[1]}_{LOSS_TYPE}_{SEED}_model_checkpoint.pth")

    if sys.argv[1] == "hybrid" and (LOSS_TYPE == "mse"or LOSS_TYPE == "mae"):
        Model = HybridModel1Out
    elif sys.argv[1] == "hybrid" and LOSS_TYPE == "nll":
        Model = HybridModel
    
    # load in pickled dataset from file and instantiate DataLoader Object
    dataset = torch.load('dataloader_2008_2025.pth', weights_only=False) # load in saved dataset


    # Split dataset 
    dataset, test_dataset = torch.utils.data.random_split(dataset, [0.90, 0.10], generator=torch.Generator().manual_seed(SEED))

    # Create DataLoader for train and test after the best model is selected
    all_train_dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # https://medium.com/biased-algorithms/cross-validation-in-pytorch-2f9f9fa9ab16
    # Initialize KFold
    kfold = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)

    # create a list of L2 penalties to test
    l2_alpha_list = [0.10, 0.01, 0.001, 0]
    # to store the average loss (over all folds) with each regularization parameter
    l2_loss_list = list()

    # Readable softmax function (percentages)
    # softmax = nn.Softmax(dim=-1)

    global loss_is_nll
    global log_softmax
    log_softmax = nn.LogSoftmax(dim=-1)
    # Initialize the loss function, currently supports MSE, MAE, and NLL
    if LOSS_TYPE == "mse":
        loss_fn = nn.MSELoss()
    elif LOSS_TYPE == "mae":
        loss_fn = nn.L1Loss()
    elif LOSS_TYPE == "nll":
        loss_is_nll = True
        loss_fn = nn.NLLLoss()

    print(f"Training the {sys.argv[1]} model")
    print(f"Using {loss_fn} as the loss function")
    print(f"Using {SEED} as the seed for splitting the dataset")
    

    model = Model().to(device)
    optimizer = optim.Adam(params=model.parameters())

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
            train_loader, val_loader = init_loaders(train_idx, val_idx, dataset)
            model = model if restarting_from_checkpoint else Model().to(device)
            optimizer = optimizer if restarting_from_checkpoint else optim.Adam(model.parameters(), lr=0.01, weight_decay=l2_alpha) 

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
    optimizer = optim.Adam(best_model.parameters(), lr=0.01, weight_decay=best_l2_val)

    # Retrain on the 90 percent of the data
    print("Retraining best model on 90 percent of the data\n")
    for epoch in range(NUM_EPOCHS):
        for batch_num, data in enumerate(all_train_dataloader):
            x_train, y_train = data

            x_train, y_train = x_train.to(device), y_train.float().to(device)

            optimizer.zero_grad() # zero the parameter gradients at the beginning
            y_hat = best_model(x_train)
            if loss_is_nll:
                y_hat = log_softmax(y_hat)
                y_train = y_train.long()

            loss = loss_fn(input=y_hat, target=y_train)
                    
            loss.backward()
            optimizer.step()

    print("Testing the retrained best model\n")

    # ----------------- TESTING -----------------
    # Evaluate the model on the test set
    running_test_loss = 0.0
    best_model.eval()

    # To store the distribution of classes
    actual_output_counts = np.zeros(10, dtype=int)
    model_output_counts = np.zeros(10, dtype=int)
    with torch.no_grad():
        for x_test, y_test in test_dataloader:
            x_test, y_test = x_test.to(device), y_test.float().to(device)
            y_hat_test = best_model(x_test)

            if loss_is_nll:
                y_hat_test = log_softmax(y_hat_test)
                y_test = y_test.long()

            test_loss = loss_fn(input=y_hat_test, target=y_test)
            running_test_loss += test_loss.item()

            probs = F.softmax(y_hat_test, dim=1)  # Get class probabilities

        for i in range(len(y_test)):
            true_label = y_test[i].item()
            predicted_class = torch.argmax(y_hat_test[i]).item()
            top_prob = probs[i][predicted_class].item()
            actual_output_counts[true_label] += 1
            model_output_counts[predicted_class] += 1

            # Calculate accuracy
            if true_label == predicted_class: 
                num_correct += 1
            if true_label == 0 and predicted_class > 1:
                num_false_positive += 1
            if true_label > 0 and predicted_class == 0:
                num_false_negative += 1

    
    print(f"For the test data, the actual distribution of classes are: {actual_output_counts}")
    print(f"For the test data, the model's distribution of classes are: {model_output_counts}")

    print(f"Accuracy is: {num_correct}/{len(test_dataset)}, or {num_correct/len(test_dataset) * 100}%")
    print(f"The false positive rate is: {num_false_positive}/{len(test_dataset)}, or {num_false_positive/len(test_dataset) * 100}%")
    print(f"The false negative rate is: {num_false_negative}/{len(test_dataset)}, or {num_false_negative/len(test_dataset) * 100}%")


    # print(f"Best model loss on validation: {min(l2_loss_list)}\n")
    # print(f"len(test_dataloader) is {len(test_dataloader)}")
    # avg_test_loss = running_test_loss / len(test_dataloader)
    # print(f"Best model loss on test: {avg_test_loss:.4f}\n")

    # Save the best model
    torch.save(best_model.state_dict(), os.path.join(output_dir, formatted_curr_date + f"_best_{sys.argv[1]}_mse_model_w_seed_{SEED}.pth"))
    # Remove the checkpoint file
    os.remove(MODEL_CHECKPOINT_PATH)

if __name__ == "__main__":
    main()
    output_file.close()
