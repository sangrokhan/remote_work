# Statistical Graph Visualization (stat_graph_vis)

다양한 통계 지표 간의 상관관계를 분석하여 네트워크 그래프 형태로 시각화하는 도구입니다.

## 데이터셋: 대학 전공별 졸업생 성과 (College Majors Dataset)

기존의 단순한 인구 통계보다 더 풍부한 변수 간 관계를 파악하기 위해, 미국 대학 전공별 졸업생의 소득, 취업률, 직업 유형 데이터를 활용합니다.

### 주요 분석 지표 (Columns)
*   **Women %**: 전공 내 여성 비율
*   **Median Salary**: 졸업생 소득 중위값
*   **Unemp Rate**: 해당 전공의 실업률
*   **Major Jobs**: 전공 관련 직업 취업자 수
*   **Non-Major Jobs**: 전공 무관 직업 취업자 수
*   **Low-Wage Jobs**: 저임금 직업 취업자 수

## 데이터 준비 (수동 다운로드)

프록시나 보안 설정으로 인해 명령어를 통한 다운로드가 제한될 수 있습니다. 아래 링크에서 데이터를 직접 다운로드하여 프로젝트 폴더에 넣어주세요.

1.  **다운로드 링크**: [recent-grads.csv (FiveThirtyEight)](https://raw.githubusercontent.com/fivethirtyeight/data/master/college-majors/recent-grads.csv)
2.  **설치 경로**: 다운로드한 파일의 이름을 **`college_recent_grads.csv`**로 변경하여 아래 경로에 저장하세요.
    *   `~/repo/remote_work/stat_graph_vis/data/college_recent_grads.csv`

## 시각화의 의미 (Graph & Edge)

이 도구는 각 지표(Column)를 **Node**로, 지표 간의 상관관계(Correlation)를 **Edge**로 표현합니다.

*   **Edge 유무**: 상관계수의 절대값이 임계값(예: 0.2)을 넘을 때 연결됩니다.
*   **Edge 색상**: 
    *   **파란색(Blue)**: 양의 상관관계 (한 지표가 증가할 때 다른 지표도 증가)
    *   **빨간색(Red)**: 음의 상관관계 (한 지표가 증가할 때 다른 지표는 감소)
*   **Edge 굵기**: 상관관계가 강할수록 선이 굵어집니다.

예를 들어, **'여성 비율(Women %)'**과 **'중위 소득(Median Salary)'** 사이에 굵은 **빨간색 선**이 있다면, 해당 데이터셋에서 여성 비율이 높은 전공일수록 소득 수준이 낮아지는 경향이 있음을 시각적으로 즉시 파악할 수 있습니다.

## 실행 방법
```bash
python3 main.py
```
실행 결과로 `major_correlation_network.png` 파일이 생성됩니다.
