
```python
import torch # 파이썬 머신러닝 오픈소스 라이브러리
from transformers import BertTokenizer, BertForSequenceClassification
from scipy.special import softmax
import pandas as pd
import numpy as np

# Step 1: BERT 모델 및 토크나이저 로드
MODEL_NAME = "nlptown/bert-base-multilingual-uncased-sentiment"  # 다국어 감성 분석 모델
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
model = BertForSequenceClassification.from_pretrained(MODEL_NAME)

# Step 2: 샘플 댓글 데이터 생성
data = {
    "댓글": [
        "곡선이 있고 손목받침대가 있어서 편합니다.",
        "신속하고 안전한 배송 감사합니다.",
        "생각보다 엄청 편한느낌이 아닌거 같아요",
        "사무실에서 쓰기에는 약간 소리가 있지만 편하고 좋아요.",
        "때가 너무 잘 타요.",
        " 적응시간이 좀 필요하지만 손목을 위해 구매했어요",
        "인생 키보드 만났네요.",
        "누름감이 부드럽고 좋아요."
        "비싸서 살까말까 망설였지만 좋아요.",
        "생각보다 금방 적응돼요.",
        "키 누르는 감이 부드럽고, 손목 받침대가 편합니다.",
        "사무실 책상이 작은 편인데 컴팩트해서 좋아요.",
        "로지텍 믿고 구입해요.",
        "익숙해지기 어렵네요.",
        "생각보다 작긴한데 손 작은 여성들이 쓰면 좋을 거 같아요.",
        "그럭저럭 괜찮습니다. 인체공학적인지는 잘 모르겠어요.",
        "만족합니다.",
        "한번 시도 해보기 나쁘지 않아요.",
        "색이 예쁘지만 업무 효율성이 좋은 키보드는 아니에요.",
        "키배열이 불편해요.",
        "충전할 필요 없이 오래 쓸 수 있는 건 장점입니다.",
        "빠른 배송 감사합니다.",
        "손목이 안아파서 너무 편합니다.",
        "손목 보호하려고 샀는데 최고입니다.완전 추천해요.",
        "깔끔하고 소음도 없어요.",
        "가격이 비싸서 망설였지만 품질이 좋아서 만족합니다.",
        "디자인도 예쁘고 사용하기 편리해요.",
        "가격대비 별로입니다."

    ]
}

df = pd.DataFrame(data)

# Step 3: 감성 분석 함수 정의
def predict_sentiment1(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    scores = softmax(outputs.logits.numpy())[0]  # 확률값 변환
    sentiment_score1 = scores.argmax()  # 가장 높은 점수의 인덱스 선택 (0~4)

    # BERT 감성 분석 모델 기준 (0: 매우 부정, 1: 부정, 2: 중립, 3: 긍정, 4: 매우 긍정)
    sentiment_labels1 = ["매우 부정", "부정", "중립", "긍정", "매우 긍정"]
    return sentiment_labels1[sentiment_score1], scores[sentiment_score1]  # 감정 라벨 및 확률값 반환

# Step 4: 댓글 감성 분석 수행
df["감정"], df["확률"] = zip(*df["댓글"].apply(predict_sentiment1))

# Step 5: 감성 분석 결과 출력
print("감정분석결과")
display(df)


#Step 6. 파이차트 시각화
import matplotlib.pyplot as plt
!pip install koreanize-matplotlib
import koreanize_matplotlib  # 한글 폰트 적용

# 감정 분석 결과 집계
sentiment_counts1 = df["감정"].value_counts().sort_values(ascending=False)


#시각화
plt.figure(figsize=(8, 6))
plt.pie(
    sentiment_counts1,
    labels=sentiment_counts1.index,
    autopct='%1.1f%%',
    colors=['grey','green','blue','red','orange'],
    startangle=140
)
plt.title("댓글 감성 분석 결과")
plt.show()
```    
