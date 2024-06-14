import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize
from tqdm import tqdm
import numpy as np
import matplotlib.font_manager as font_manager
# 폰트 경로 설정


#그래프 한국어 오류 막기 위함
# 'malgun.ttf'는 '맑은 고딕' 폰트의 파일 경로
font_path = 'C:/Windows/Fonts/malgun.ttf'  

# 폰트 이름 찾기
font_name = font_manager.FontProperties(fname=font_path).get_name()

# matplotlib의 폰트를 설정
plt.rcParams['font.family'] = font_name
#test위한 함수


def test_and_visualize(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    all_labels = []
    all_predictions = []
    all_probabilities = []

    with torch.no_grad():
        for images, labels, weights in tqdm(loader, desc="Testing"):
            images = torch.stack([img.float() for img in images], dim=1).to(device)
            labels = labels.to(device)
            weights = weights.float().to(device)

            outputs = model(images, weights)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * labels.size(0)
            predictions = outputs.argmax(1)
            probabilities = torch.softmax(outputs, dim=1)
            total_correct += (predictions == labels).sum().item()
            total_samples += labels.size(0)

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    
    # F1-score 계산
    f1 = f1_score(all_labels, all_predictions, average='weighted')
    
    # AUC score 계산
    all_labels_bin = label_binarize(all_labels, classes=[0, 1, 2])
    auc_score = roc_auc_score(all_labels_bin, all_probabilities, multi_class='ovr')

    print(f"Average Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}, F1-score: {f1:.4f}, AUC: {auc_score:.4f}")

    # loss, Accuracy, F1-score, AUC score 차트로 시각화
    plt.figure(figsize=(12, 5))
    metrics = ['Average Loss', 'Accuracy', 'F1-score', 'AUC']
    values = [avg_loss, accuracy, f1, auc_score]
    plt.bar(metrics, values, color=['blue', 'green', 'red', 'purple'])
    plt.ylabel('Values')
    plt.title('Test Metrics')
    plt.ylim(0, 1.0)
    for i, v in enumerate(values):
        plt.text(i, v + 0.05, f"{v:.4f}", ha='center', color='black')
    plt.show()

    # confusion_matrix 시각화
    cm = confusion_matrix(all_labels, all_predictions)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['특', '상', '보통'], yticklabels=['특', '상', '보통'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

    # ROC 커브 시각화
    plt.figure(figsize=(10, 7))
    for i in range(len(np.unique(all_labels))):
        fpr, tpr, _ = roc_curve(all_labels_bin[:, i], np.array(all_probabilities)[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'Class {i} (area = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.show()