# Statistical Graph Visualization (stat_graph_vis)

통계 지표 간의 상관관계를 분석하여 네트워크 그래프 형태로 시각화하는 도구입니다. 이 프로젝트는 오프라인 환경에서도 구동 가능하도록 설계되었습니다.

## 데이터 준비 (수동 다운로드)

프록시나 보안 설정으로 인해 명령어를 통한 다운로드가 제한될 수 있습니다. 아래 링크에서 데이터를 직접 다운로드하여 프로젝트 폴더에 넣어주세요.

1.  **데이터셋**: Gapminder (Life Expectancy, GDP, Population)
2.  **다운로드 링크**: [gapminderDataFiveYear.csv](https://raw.githubusercontent.com/plotly/datasets/master/gapminderDataFiveYear.csv)
3.  **설치 경로**: 다운로드한 파일의 이름을 `gapminder_combined.csv`로 변경하여 아래 경로에 저장하세요.
    *   `~/repo/remote_work/stat_graph_vis/data/gapminder_combined.csv`

## 설치 및 실행

### 필요 라이브러리
본 프로젝트는 아래 라이브러리들을 사용합니다 (오프라인 설치 필요 시 해당 패키지들을 미리 준비하세요).
*   `pandas`
*   `networkx`
*   `matplotlib`

### 실행 방법
데이터가 준비된 후 아래 명령어로 시각화를 수행합니다.
```bash
python3 main.py
```
실행 결과로 `gapminder_network.png` 파일이 생성됩니다.

## 시각화 로직
1.  **데이터 로드**: `pandas`를 이용해 Gapminder 통계 데이터를 읽어옵니다.
2.  **상관분석**: 인구수, 기대수명, 1인당 GDP 지표 간의 피어슨 상관계수를 계산합니다.
3.  **네트워크 생성**: 상관계수의 절대값이 임계값(기본 0.3)을 넘는 지표들을 `Edge`로 연결합니다.
4.  **시각화**: `NetworkX`와 `Matplotlib`을 사용하여 로컬 환경에서 그래프 이미지를 생성합니다.
