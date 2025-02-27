```python
import matplotlib.pyplot as plt
import pandas as pd

# 데이터 합치기
sentiment_df = pd.DataFrame({
    "로지텍": sentiment_counts1,
    "키크론": sentiment_counts2
}).fillna(0)  # 감정이 없는 경우 0으로 채우기

# 원하는 감정 순서 지정
emotion_order = ["매우 부정", "부정", "중립", "긍정", "매우 긍정"]

# 데이터프레임 재정렬
sentiment_df = sentiment_df.reindex(emotion_order)

# 막대 그래프 그리기
sentiment_df.plot(kind="bar", figsize=(10, 6), color=["yellow", "purple"], alpha=0.7)

# 그래프 설정
plt.title("로지텍과 키크론 댓글 감성 분석 비교")
plt.xlabel("감정")
plt.ylabel("댓글 수")
plt.xticks(rotation=0)  # X축 글자 가로 정렬
plt.legend(title="브랜드")  # 범례 추가

# 그래프 출력
plt.show()
```
