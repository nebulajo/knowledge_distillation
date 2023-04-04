# Knowledge Distillation

paper : Distilling the Knowledge in a Neural Network

앙상블된 모델 또는 규모가 더 큰 모델(파라미터 개수가 많은)의 지식을 증류하는 방법이다. 즉 pre-trained 모델이 학습한 feature를 학습하는 training 기법이다. 이때 pre-trained 모델을 teacher model, 해당 모델의 지식을 학습하는 모델을 student model로 정의한다.

Knowledge distillation에서는 softmax 함수 결과를 그대로 사용한다. 일반적으로 분류 모델을 학습할 때 softmax 함수의 결과로 각 클래스에 대한 확률값이 출력되지만 이후 가장 높은 확률값의 index를 1, 그 외에는 0으로 처리하게 된다. 논문에서는 0으로 처리되는 클래스 확률 자체도 의미가 있기 때문에 이를 같이 학습할 수 있도록 구성한다.

<img width="706" alt="스크린샷 2023-04-04 오후 10 31 29" src="https://user-images.githubusercontent.com/126544082/229809213-c77638d6-ea0c-4d18-843c-68108a7ca277.png">
Ref : https://intellabs.github.io/distiller/knowledge_distillation.html

### Distillation

<img width="260" alt="스크린샷 2023-04-04 오후 7 55 34" src="https://user-images.githubusercontent.com/126544082/229809231-0b8506de-004f-4467-8c86-be0229f4ea61.png">

Teacher model의 softmax 값을 T(Temperature)로 나눠서 softmax를 다시 한 번 취한다. **T 값이 높아질 수록 출력값이 soft 해지는데 이는 출력 분포의 형태가 뾰족하지 않고 점차 균등한 것을 의미한다.** 원 핫 벡터 형태가 아니더라도 특정 클래스의 확률이 차이나게 높을 수 있기 때문에 이를 보완할 수 있다. 논문에서는 웟 핫 벡터 형식의 label을 hard label, softmax를 거친 확률값 형태를 soft label이라고 정의한다.
* soft label : softmax 결과, 확률 값 ex) [0.10, 0.20, 0.70]
* hard label : softmax 결과에서 원 핫 벡터 형식으로 바꾼 것 ex) [0, 0, 1]

### objective function

여기서 objective function은 loss function으로 이해해도 된다. knowledge distillation을 구현하기 위해서는 loss function이 크게 두 가지로 구성되며 다음과 같다.

* first objective(distillation loss) : teacher model과 student model 간의 soft label 차이(KLD로 구현) 
* second objective(student loss) : student model과 label 간의 hard label 차이

두 loss function에 대해서 weighted sum한 결과가 가장 성능이 좋다고 한다. 

```python
# knowledge distillation loss
def distillation_loss(student_scores, targets, teacher_scores, T, alpha):
    # distillation loss + classification loss
    # student_scores : student model ouputs (soft label) 
    # targets : labels
    # teacher_scores: teacher model outputs (soft label)

    distillation_loss = nn.KLDivLoss()(F.log_softmax(student_scores/T), F.softmax(teacher_scores/T))  
    student_loss = F.cross_entropy(student_scores,targets) 

    # distillation_loss, student_loss의 weighted sum으로 계산
    return distillation_loss*(T*T * 2.0 + alpha) + loss2*(1.-alpha)
```

## Experiment
본 실험은 CIFAR10 데이터셋으로 진행하였다. ResNet34의 경우 85% 가량의 test acc가 나온다. 본 실험에서는 Augmentation을 추가로 하지 않고, Knowledge Distillation을 통해 CIFAR10 데이터셋에 대해서 ResNet38 모델의 test acc를 85%보다 더 높은 성능이 나타나도록 실험을 진행한다. Teacher Model의 경우 ImageNet으로 사전학습된 ViT(vit_base_patch16_224)으로 구성하고, Student Model의 경우 ImageNet으로 사전학습된 ResNet38 모델로 구성한다. 

- Teacher Model : ViT(vit_base_patch16_224) pretrained by ImageNet 
    - Param : 86M
- Student Model : ResNet34 pretrained by ImageNet
    - Param : 21.5M

### ViT Finetuning

CIFAR10 데이터셋에 대해서 ViT를 Teacher Model로 사용하기 위해서 ViT의 마지막 FC layer의 출력층을 CIFAR10 데이터셋의 클래스 수인 10으로 맞춰줘야 한다.  `nn.Linear(512, 10)` 을 적용하게 되면 해당 레이어의 가중치는 랜덤한 초기 가중치이고, 이는 Student Model(ResNet38)에게 ViT의 ImageNet feature를 전달하는 데에 방해가 될 수 있다고 판단하였다. 따라서 본 실험에서는 nn.Linear에 대해서만 requires_grad=True로 설정해서 finetuining을 진행한 이후 Teacher Model로 사용하였다.

### reference
* https://deep-learning-study.tistory.com/700
* https://github.com/ndb796
