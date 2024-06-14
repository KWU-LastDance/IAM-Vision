import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torchvision
import torchvision.transforms.functional as F
from PIL import Image
import io
from itertools import cycle
import numpy as np
from sklearn.preprocessing import label_binarize
from sklearn.metrics import confusion_matrix,roc_auc_score, f1_score,roc_curve, auc
import wandb

# 학습 평가기준인 confusion matrix구하기
def plot_confusion_matrix(cm, class_names,phase,epoch):
    figure = plt.figure(figsize=(8, 8))
    sns.heatmap(cm, annot=True, cmap=plt.cm.Blues, xticklabels=class_names, yticklabels=class_names, fmt='g')
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.title(f'Confusion Matrix: {phase}_Epoch {epoch+1}')
    return figure

def save_confusion_matrix(y_true, y_pred, class_names, phase,epoch):
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_figure = plot_confusion_matrix(cm, class_names,phase,epoch)
    cm_normalized_figure = plot_confusion_matrix(cm_normalized, class_names,phase,epoch)
    wandb.log({f"{phase}_confusion_matrix": wandb.Image(cm_figure)})
    wandb.log({f"{phase}_confusion_matrix_normalized": wandb.Image(cm_normalized_figure)})



#학습 결과 추론(이미지 분류 확인)

def unnormalize(tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    tensor = tensor * std + mean  
    tensor = torch.clamp(tensor, 0, 1)  
    return tensor

def log_grouped_images_to_wandb(images, outputs, labels, epoch, device,mode):
    predictions = outputs.argmax(1)
    batch_size = labels.size(0)

    for idx in range(batch_size):
        # 배치의 각 샘플에 대하여
        sample_images = images[idx]  # [num_angles, channels, height, width] 형태일 것
        sample_images = [unnormalize(sample_images[angle]) for angle in range(sample_images.shape[0])]
        image_grid = torchvision.utils.make_grid(torch.stack(sample_images), nrow=6, padding=2, normalize=False)
        if(mode=='train'):
            # wandb에 그리드 로깅
            wandb.log({f"Train End of Epoch {epoch}": [wandb.Image(image_grid.cpu().permute(1, 2, 0).numpy(),
                                                             caption=f"Predicted: {predictions[idx]}, Actual: {labels[idx]}")]})
        else:
            wandb.log({f"Validation End of Epoch {epoch}": [wandb.Image(image_grid.cpu().permute(1, 2, 0).numpy(),
                                                             caption=f"Predicted: {predictions[idx]}, Actual: {labels[idx]}")]})
            
            



#성능지표인 roccurve 구하기
def plot_roc_curves_to_wandb(labels, probs, classes, mode, epoch):
    labels_bin = label_binarize(labels, classes=classes)
    n_classes = len(classes)

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(labels_bin[:, i], probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure()
    colors = cycle(['blue', 'red', 'green'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label='ROC curve of class {0} (area = {1:0.2f})'.format(i, roc_auc[i]))
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    
    plt.title(f'Multi-class ROC ({mode}, Epoch {epoch})')
    plt.legend(loc="lower right")
    plt.grid(True)

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    image = Image.open(buf)
    wandb.log({f"{mode} ROC Curves": wandb.Image(image, caption=f"{mode} ROC Curves, Epoch {epoch}")})


