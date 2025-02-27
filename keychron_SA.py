```python
import torch # 파이썬 머신러닝 오픈소스 라이브러리
from transformers import BertTokenizer, BertForSequenceClassification
from scipy.special import softmax
import pandas as pd

# Step 1: BERT 모델 및 토크나이저 로드
MODEL_NAME = "nlptown/bert-base-multilingual-uncased-sentiment"  # 다국어 감성 분석 모델
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
model = BertForSequenceClassification.from_pretrained(MODEL_NAME)

# Step 2: 샘플 댓글 데이터 생성
data = {

    "댓글": [
        "만족하며 사용하고 있습니다.",
        "적응에 시간이 필요해요.",
        "색상이 예쁘다는 이점이 있지만 적응에 시간이 필요해요.",
        "제품이 마음에 들어요.",
        "배송도 빠르고 잘 쓰고 있습니다.",
        "금방 도착했어요.",
        "소리가 좋고 크기도 적당해서 좋아요.",
        "돈 값어치 하네요.",
        "자판도 잘 눌리고 마음에 들어요.",
        "키보드 반응속도도 빠르고 매우 만족합니다.",
        "아주 좋습니다.",
        "비싼 만큼 성능이 좋아요.",
        "한달동안 하루도 빠지지 않고 게임을 하는데 너무 좋아요.",
        "기대 이상입니다.",
        " 최고인거 같아요.",
        "as 가능해서 좋아요.",
        "만족하며 사용하고 있습니다.",
        "키보드 루프 증정이 너무 더러웠지만 제품이 마음에 들어요.",
        "비싸요.",
        "3년 쓰니 고장나네요.",
        "소음이 살짝 있지만 괜찮아요.",
        "이 돈 주고 살 정도는 아니에요.",
        "강추합니다.",
        "대박입니다.",
        "다른 색상도 있으면 좋겠어요.",
        "처음 쓰는데 좋습니다.",
        "가격때문에 망설였어요."
    ]
}

df = pd.DataFrame(data)

# Step 3: 감성 분석 함수 정의
def predict_sentiment2(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    scores = softmax(outputs.logits.numpy())[0]  # 확률값 변환
    sentiment_score2 = scores.argmax()  # 가장 높은 점수의 인덱스 선택 (0~4)


    # BERT 감성 분석 모델 기준 (0: 매우 부정, 1: 부정, 2: 중립, 3: 긍정, 4: 매우 긍정)
    sentiment_labels2 = ["매우 부정", "부정", "중립", "긍정", "매우 긍정"]
    return sentiment_labels2[sentiment_score2], scores[sentiment_score2]  # 감정 라벨 및 확률값 반환

# Step 4: 댓글 감성 분석 수행
df["감정"], df["확률"] = zip(*df["댓글"].apply(predict_sentiment2))

# Step 5: 감성 분석 결과 출력
print("감정분석결과")
display(df)

#Step 6. 파이차트 시각화
import matplotlib.pyplot as plt
!pip install koreanize-matplotlib
import koreanize_matplotlib  # 한글 폰트 적용

# 감정 분석 결과 집계
sentiment_counts2 = df["감정"].value_counts().sort_values(ascending=False)


# 시각화
plt.figure(figsize=(8, 6))
plt.pie(
    sentiment_counts2,
    labels=sentiment_counts2.index,
    autopct='%1.1f%%',
    colors=['grey','green','blue','red','orange'], # 각도지점에서 반시계방향으로 색 지정 됨.
    startangle=140
)
plt.title("댓글 감성 분석 결과")
plt.show()
```
