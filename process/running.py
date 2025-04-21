import torch

from sklearn.metrics import precision_recall_fscore_support

def binary_accuracy(y_pred, y_test):
    threshold = 0.50  # Replace with your desired threshold value
    y_pred_tag = (y_pred >= threshold).float()
    correct_results = (y_pred_tag == y_test).sum().float()
    acc = correct_results/y_test.shape[0]
    return acc,y_pred_tag

def train(model,train_loader,criterion,optimizer,device):
    model.train()
    epoch_loss = 0
    epoch_acc = 0
    
    for input_data in train_loader:
        input_data = input_data.to(device)
        y_batch = torch.tensor(input_data.y, dtype=torch.float32).view(-1, 1)
        
        # Forward pass
        y_pred = model(input_data)
        loss = criterion(y_pred, y_batch)
        acc,_ = binary_accuracy(y_pred, y_batch)
        
        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_acc += acc.item()
    
    return model, epoch_loss, epoch_acc

def test(model,test_loader,criterion,device):
    model.eval()
    val_losses = []

    val_accs = []

    target = []
    pred = []
    y_probs = []
    with torch.no_grad():
        for input_data_test in test_loader:
            input_data_test = input_data_test.to(device)
            y_batch = torch.tensor(input_data_test.y, dtype=torch.float32).view(-1, 1)
            target += input_data_test.y
            
            y_pred = model(input_data_test)
            loss = criterion(y_pred, y_batch)
            
            acc,pred_target = binary_accuracy(y_pred, y_batch)
            pred += pred_target.float().squeeze().tolist()
            
            val_losses.append(loss.item())
            val_accs.append(acc.item())

            y_probs.append(y_pred)


    precision,recall,fscore,_ = precision_recall_fscore_support(target,pred,average="binary")

    # y_probs = torch.cat(y_probs, dim=0).squeeze().numpy()
    # y_target = np.array(target)

    # threshold_plot(y_probs,y_target)
    
    return val_losses, val_accs, precision, recall, fscore

def test_with_prediction(model,test_loader,criterion,device):
    model.eval()
    val_losses = []
    val_accs = []
    target = []
    pred = []
    y_probs = []
    pred_labels = []
    true_labels = []

    with torch.no_grad():
        for input_data_test in test_loader:
            input_data_test = input_data_test.to(device)
            y_batch = torch.tensor(input_data_test.y, dtype=torch.float32).view(-1, 1)
            target += input_data_test.y
            true_labels.extend(input_data_test.y)
            
            y_pred = model(input_data_test)
            loss = criterion(y_pred, y_batch)
            
            acc,pred_target = binary_accuracy(y_pred, y_batch)
            pred += pred_target.float().squeeze().tolist()
            pred_labels.extend(pred_target.float().squeeze().tolist())
            
            val_losses.append(loss.item())
            val_accs.append(acc.item())
            y_probs.append(y_pred)


    precision,recall,fscore,_ = precision_recall_fscore_support(target,pred,average="binary")

    # y_probs = torch.cat(y_probs, dim=0).squeeze().numpy()
    # y_target = np.array(target)

    # threshold_plot(y_probs,y_target)
    
    return val_losses, val_accs, precision, recall, fscore, true_labels, pred_labels