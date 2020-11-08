# DecisionTreeScratch
공부하기 위해 만든 간단한 decision tree 코드.

가장 간단한 binary tree regression 을 구현한 simple_tree.py,

bagging을 구현한 random_forest.py,

boosting(gradient, adaptive)을 구현한 dbt.py,

등으로 구성됨.

$python DesicionTreeClassifier.py

명령어를 통해 실행 가능함.

# Dataset

data_banknote_authentication.csv

총 5개의 열(column)로 구성. 마지막 열은 label로 0 또는 1

https://archive.ics.uci.edu/ml/datasets/banknote+authentication

# Binary tree

node와 leaf로 구성. node에는 분류 조건( x > value )이 포함되고 자식으로 2개의 node 또는 leaf를 가질 수 있다.

leaf에는 예측시 출력될 상수 값이 들어간다.

1, 0 같은 정수, 혹은 분류된 목표값들의 평균 등이 들어갈 수 있다.

너무 많은 패턴이 학습되는 것을 막기 위해 보통 제약조건을 넣어준다.

최대 깊이(depth)를 정하여 하위 node의 생성을 제한하거나

leaf에 분류된 데이터의 최솟값을 제한하여 지엽적인 패턴이 학습되는 것을 막을 수 있다.

(leaf의 개수를 제한하는 방법도 사용 가능함.)

# Ensemble learning
여러 개의 트리를 만들어 예측값의 평균, 중앙값(median), 가중 평균을 사용하는 방법.

## bagging

training 샘플중 일부를 임의로 샘플링하여(50%-80%가 적당함) 트리를 반복적으로 생성.

생성된 모든 트리의 예측값을 평균내어 사용.

각 트리가 독립적으로 생성되어 과적합(overfitting) 개선.

## boosting
이전 트리들이 학습하지 못한 패턴에 집중하여 새로운 트리를 생성하고,

생성된 모든 트리들의 예측값을 가중 평균하여 사용.

이전 트리들에 지속적으로 새로운 트리를 더해 개선된 모델을 만들어 나가는 방식.
