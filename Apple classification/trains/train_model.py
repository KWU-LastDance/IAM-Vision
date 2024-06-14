import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score, roc_curve
import wandb
from tqdm import tqdm
from sklearn.preprocessing import label_binarize
from Evaluation_Metrics import save_confusion_matrix,log_grouped_images_to_wandb, plot_roc_curves_to_wandb


#학습하기
def train(model, loader, optimizer, criterion, device, epoch):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    all_labels = []
    all_predictions = []
    all_probs = []

    for step, (images, labels, weights) in enumerate(tqdm(loader, desc="Training")):
        batch_size, num_views, channels, height, width = images.shape
        images_list = [images[:, i, :, :, :].to(device) for i in range(num_views)]
        labels = labels.to(device)
        weights = weights.float().to(device)

        optimizer.zero_grad()
        outputs = model(images_list, weights)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        total_correct += (outputs.argmax(1) == labels).sum().item()
        total_samples += labels.size(0)

        all_labels.extend(labels.cpu().numpy())
        all_predictions.extend(outputs.argmax(1).cpu().numpy())
        probs = torch.nn.functional.softmax(outputs, dim=1).detach()
        all_probs.append(probs.cpu().numpy())

        if step < 1:  # 첫 배치만 (결과 추론 이미지)
            log_grouped_images_to_wandb(images, outputs, labels, epoch, device, mode='train')

        
        #빠른 계산을 위해
        del images_list, labels, weights, outputs, loss
        torch.cuda.empty_cache()

    all_probs = np.concatenate(all_probs, axis=0)
    epoch_loss = total_loss / total_samples
    epoch_accuracy = total_correct / total_samples
    
    #평가지표 f1 Score, auc Score, confusion_matrix, Roc curve, loss, accuracy를 wandb로 보냄
    f1 = f1_score(all_labels, all_predictions, average='weighted')
    auc_score = roc_auc_score(label_binarize(all_labels, classes=[0, 1, 2]), all_probs, multi_class='ovr')
    
    save_confusion_matrix(all_labels, all_predictions, class_names=['Class 0', 'Class 1', 'Class 2'], phase="train",epoch=epoch)

    wandb.log({"Train Loss": epoch_loss, "Train Accuracy": epoch_accuracy, "Train F1 Score": f1, "Train AUC": auc_score})

    plot_roc_curves_to_wandb(all_labels, all_probs, [0, 1, 2], mode='train', epoch=epoch)

    return epoch_loss, epoch_accuracy


#validation 

def validate(model, loader, criterion, device, epoch):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    all_labels = []
    all_predictions = []
    all_probs = []

    with torch.no_grad():
        for step, (images, labels, weights) in enumerate(tqdm(loader, desc="Validating")):
            batch_size, num_views, channels, height, width = images.shape
            images_list = [images[:, i, :, :, :].to(device) for i in range(num_views)]
            labels = labels.to(device)
            weights = weights.float().to(device)

            outputs = model(images_list, weights)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * labels.size(0)
            total_correct += (outputs.argmax(1) == labels).sum().item()
            total_samples += labels.size(0)

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(outputs.argmax(1).cpu().numpy())
            probs = torch.nn.functional.softmax(outputs, dim=1).detach().cpu().numpy()
            all_probs.append(probs)

            if step < 1:  # # 첫 배치만 (결과 추론 이미지)
                log_grouped_images_to_wandb(images, outputs, labels, epoch, device, mode='validation')
            

        all_probs = np.concatenate(all_probs, axis=0)
        epoch_loss = total_loss / total_samples
        epoch_accuracy = total_correct / total_samples
        
        #평가지표 f1 Score, auc Score, confusion_matrix, Roc curve, loss, accuracy를 wandb로 보냄
        f1 = f1_score(all_labels, all_predictions, average='weighted')
        
        auc_score = roc_auc_score(label_binarize(all_labels, classes=[0, 1, 2]), all_probs, multi_class='ovr')

        wandb.log({
            "Validation Loss": epoch_loss,
            "Validation Accuracy": epoch_accuracy,
            "Validation F1 Score": f1,
            "Validation AUC": auc_score
        })
        save_confusion_matrix(all_labels, all_predictions, class_names=['Class 0', 'Class 1', 'Class 2'], phase="Validation",epoch=epoch)
        
        plot_roc_curves_to_wandb(all_labels, all_probs, [0, 1, 2], mode='Validation', epoch=epoch)

    return epoch_loss, epoch_accuracy


# 모델 저장하기
def save_model(model, save_path):
    torch.save(model.state_dict(), save_path)

#실제 학습하기


def run(config, model, optimizer,scheduler, criterion, device, train_loader, val_loader,best_accuracy,path): #path는 실제 파일이름 지정
    wandb.init(project="apple-classification-project", config=config)
    config = wandb.config

    num_epochs = config['epochs']
    patience = config['patience']
    patience_counter = 0
    best_accuracy = 0.0

    for epoch in range(num_epochs):
        train_loss, train_accuracy = train(model, train_loader, optimizer, criterion, device, epoch)
        val_loss, val_accuracy = validate(model, val_loader, criterion, device, epoch)
        print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
        
        scheduler.step()

        wandb.log({"Epoch": epoch+1, "Train Loss": train_loss, "Train Accuracy": train_accuracy, "Validation Loss": val_loss, "Validation Accuracy": val_accuracy})

        if val_accuracy >= best_accuracy:
            best_accuracy = val_accuracy
            save_model(model, f'./{path}/{epoch+1}_{best_accuracy}best_model.pth')
            print(f"Saved best model with accuracy: {best_accuracy:.4f}")
            patience_counter = 0
        elif epoch==49:
            save_model(model, f'./{path}/{epoch+1}_{best_accuracy}best_model.pth')
            print(f"Saved best model with accuracy: {best_accuracy:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {patience} epochs with no improvement")
                break
        
        #빠른 계산을 위해
        torch.cuda.empty_cache()
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        

    wandb.finish()