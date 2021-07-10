

1. voice 칼럼 유무로 파일 구분
2. eye inspection
   1. live -> 일부 1명만 있는 항목들 다른 항목으로 이동
   2. through -> `우리경민잉`, `예진이 최고` -> 지인소개
   3. domain -> `스타트업 운영` -> 자영업
   4. transportation -> 아래 3개 row 삭제
   ```
   버스와 지하철(복수응답이 안되어) | 날씨, 미세먼지, 뉴스, 코로나 관련 소식 및 확진자 수, 이동수단에 대한 정보 | 매우 의향이 있음
   버스와지하철 | 날씨, 미세먼지, 코로나 관련 소식 및 확진자 수, 이동수단에 대한 정보 | 의향 있음
   버스&지하철 | 날씨, 주식 및 비트코인, 코로나 관련 소식 및 확진자 수, 이동수단에 대한 정보 | 보통
   ```
   5. most_app
   ```
   날씨
   미세먼지
   주식 및 비트코인
   뉴스
   코로나 관련 소식 및 확진자 수
   이동수단에 대한 정보
   ---
   Sns
   유튜브, 웹툰
   오늘의 계획 -> 캘린더
   에브리타임 -> SNS
   캘린더
   게임
   과제,켈린더 -> 캘린더,에타 -> SNS
   금융 앱(토스) -> 토스
   축구, 인터넷방송
   패션
   유튜브? -> 유튜브
   영화
   업무용 SNS
   ```
   6. likes
    - 매우 의향 없음_어플 기획의도 자체를 이해하지 못함 -> 매우 의향 없음
    - 음성보단 눈으로 확인하는게 더 직관적이고 빠르게 보이는데 음성은 전달력이 좀 떨어지지 않을까 생각됩니당.. 막상 써보면 또 다를 수도 있을 것 같아서 일단 사용해보곤 싶습니다. -> 의향 있음
    - 음성이 나오는 건 좀 그렇고 푸시알림으로 뜨는 정도면 괜찮음 -> 보통
    - 저는 개인적으로 알림 알려주는게 시끄러워서 싫습니다 -> 의향 없음
    - 음성을 계속 듣고있지 않기도 하고 깜짝놀랄거같기도함 -> 의향 없음
    - 일상생활 하다가 주변에 사람들 있는데 갑자기 소리나면 민망할 것 같아요 -> 의향 없음
    - 위 과정 거친 후, 등간척도로 수정
       - 매우 의향 있음: 5 ~ 매우 의향 없음: 1
   7. age
    - 각 연령대별 평균값으로 변경

3. Preprocessing
   1. `df.info()`
   ```
   <class 'pandas.core.frame.DataFrame'>
   RangeIndex: 267 entries, 0 to 266
   Data columns (total 8 columns):
   #   Column          Non-Null Count  Dtype 
   ---  ------          --------------  ----- 
   0   age             267 non-null    object
   1   live            267 non-null    object
   2   through         266 non-null    object
   3   sex             267 non-null    object
   4   domain          263 non-null    object
   5   transportation  267 non-null    object
   6   most_app        267 non-null    object
   7   likes           267 non-null    object
   dtypes: object(8)
   memory usage: 16.8+ KB
   None
   ```
   2. `df.describe()`
   ```
                       age          live through  sex domain transportation         most_app  likes
   count               267           267     266  267    263            267              267    267
   unique                6             6       5    2      5              8               61      5
   top     20대 초반 (20-23세)  가족과 함께 살고 있음   에브리타임    여     학생            지하철  날씨, 이동수단에 대한 정보  의향 있음
   freq                185           183     171  161    218            122               42     98
   ```
   3. `df.isna().sum()`
   ```
   age               0
   live              0
   through           1
   sex               0
   domain            4
   transportation    0
   most_app          0
   likes             0
   dtype: int64
   ```
   through -> 지인 소개
   domain -> 무직
   4. likes, most_app 제외 label encoder
   5. most_app -> , 분류 후 onehot encoder


결과 #1
 - 22.0세 / 여자(65%) / 3.74 / 날씨, 이동수단, 뉴스, 비트코인, 미세먼지
 - 30.8세 / 남자(64%) / 3.56 / 날씨, 비트코인, 뉴스, 미세먼지
 - 22.2세 / 여자(64%) / 3.45 / 날씨 ... 이동수단, 뉴스, 비트코인, 미세먼지


결과 #2
 - k-Means 및 eye-inspection 사용하여 분석
 - 자세한 내용은 팀 드롭박스 참고