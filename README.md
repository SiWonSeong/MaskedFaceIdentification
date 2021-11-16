## MaskedFaceIdentification - 마스크 착용에서의 얼굴 인식 시스템

### 1. 원본 데이터
- 마스크 착용하지 않은 얼굴, 마스크 착용한 얼굴

### 2. 전처리
- 얼굴 이미지 입력
- 랜드마크 검출(68개)
- 랜드마크로 얼굴 전체(face) 또는 눈 주위 영상(periocular) 취득 (이미지 crop)

<img src="https://user-images.githubusercontent.com/70686586/141952797-4b009f4f-7ac7-4d82-b0c9-faec7feb966d.png" width="80%" height="80%"></img><br/>

### 3. 모델 학습
- Siamese Network 기반의 모델로 학습 → 얼굴 데이터를 crop하여 사용
- 105×105×1 이미지 2장 → 출력: 길이 100의 벡터 2개 → L2 distance 계산
  
##### 1. 마스크 안 쓴 face로 학습 → 마스크 안 쓴 face로 테스트 → acc 0.930 / f1 0.930
##### 2. 마스크 안 쓴 face로 학습 → 마스크 쓴 face로 테스트 → acc 0.634 / f1 0.691
##### 3. 마스크 안 쓴 periocular로 학습 → 마스크 안 쓴 periocular로 테스트 → acc 0.923 / f1 0.924
##### 4. 마스크 안 쓴 periocular로 학습 → 마스크 쓴 periocular로 테스트 → acc 0.663 / f1 0.707 ★

- 2번과 4번을 비교하였을 때 성능이 향상됨.

- 1, 2를 학습한 모델 → Siamese-face / 3, 4를 학습한 모델 → Siamese-periocular → 프로그램 제작 시 Siamese-periocular 모델을 사용

<img src="https://user-images.githubusercontent.com/70686586/141953474-add4193e-443a-42d9-ade0-484bae583c0b.png" width="80%" height="80%"></img><br/>

### 4. 모델 테스트
- 서로 다른 이미지를 1:1로 비교하여 L2 distance 계산
- f1 score가 최대가 되는 distance를 threshold로 정의
- 이러한 threshold를 기준으로 genuine pair, imposter pair를 예측하여 accuracy, f1 score, AUC를 기록함.

### 5. 참고자료
- **Seong, S. W.**, Han, N. Y., Ryu, J., Hwang, H., Joung, J., Lee, J., & Lee, E. C. (2020, November). Authentication of Facial Images with Masks Using Periocular Biometrics. In International Conference on Intelligent Human Computer Interaction (pp. 326-334). Springer, Cham.
- https://github.com/HanNayeoniee/Masked-Face-Authentication
