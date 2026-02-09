import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import os

class StatGraphVis:
    def __init__(self, data_path):
        self.data_path = data_path
        self.df = None
        self.corr_matrix = None
        self.graph = None

    def load_data(self):
        """데이터 로드 및 기본 전처리"""
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data file not found at {self.data_path}")
        
        # 전공별 졸업생 데이터 로드
        self.df = pd.read_csv(self.data_path)
        return self.df

    def calculate_correlations(self):
        """수치형 지표 간의 상관관계 계산"""
        if self.df is None:
            self.load_data()
        
        # 분석에 의미 있는 수치형 컬럼 선택
        # ShareWomen: 여성 비율, Unemployment_rate: 실업률, Median: 소득 중위값, 
        # College_jobs: 전공 관련 직업, Non_college_jobs: 전공 무관 직업, Low_wage_jobs: 저임금 직업
        cols = [
            'ShareWomen', 'Unemployment_rate', 'Median', 
            'College_jobs', 'Non_college_jobs', 'Low_wage_jobs'
        ]
        
        # 결측치 제거
        numeric_df = self.df[cols].dropna()
        
        # 한글/영문 가독성을 위해 컬럼명 변경
        numeric_df.columns = [
            'Women %', 'Unemp Rate', 'Median Salary', 
            'Major Jobs', 'Non-Major Jobs', 'Low-Wage Jobs'
        ]
        
        self.corr_matrix = numeric_df.corr()
        return self.corr_matrix

    def build_graph(self, threshold=0.3):
        """상관관계를 기반으로 그래프(네트워크) 생성"""
        if self.corr_matrix is None:
            self.calculate_correlations()
            
        self.graph = nx.Graph()
        
        nodes = self.corr_matrix.columns
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                weight = self.corr_matrix.iloc[i, j]
                # 상관계수의 절대값이 임계값을 넘는 경우만 에지로 연결
                if abs(weight) >= threshold:
                    self.graph.add_edge(nodes[i], nodes[j], weight=weight)
        
        return self.graph

    def visualize_graph(self, output_path='major_correlation_network.png'):
        """그래프 시각화 및 로컬 저장"""
        if self.graph is None:
            self.build_graph()
            
        plt.figure(figsize=(12, 10))
        # 노드가 적으므로 circular layout이 관계 파악에 용이할 수 있음
        pos = nx.circular_layout(self.graph)
        
        # 에지 속성 설정
        edges = self.graph.edges(data=True)
        if not edges:
            print("No correlations found above threshold.")
            return None
            
        weights = [abs(d['weight']) * 8 for u, v, d in edges]
        # 양의 상관관계는 파란색, 음의 상관관계는 빨간색
        edge_colors = ['red' if d['weight'] < 0 else 'blue' for u, v, d in edges]
        
        # 노드 그리기
        nx.draw_networkx_nodes(self.graph, pos, node_size=3000, node_color='lightgray', edgecolors='black')
        
        # 에지 그리기
        nx.draw_networkx_edges(self.graph, pos, width=weights, edge_color=edge_colors, alpha=0.6)
        
        # 라벨 그리기
        nx.draw_networkx_labels(self.graph, pos, font_size=10, font_weight='bold')
        
        # 범례 대신 텍스트 추가
        plt.text(1.1, 1.1, "Blue: Positive Corr\nRed: Negative Corr\nThickness: Strength", 
                 transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.5))

        plt.title("Correlation Network between College Major Outcomes", pad=20)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        return output_path

if __name__ == "__main__":
    data_file = os.path.expanduser("~/repo/remote_work/stat_graph_vis/data/college_recent_grads.csv")
    vis = StatGraphVis(data_file)
    try:
        vis.load_data()
        vis.calculate_correlations()
        vis.build_graph(threshold=0.2) # 관계를 더 많이 보기 위해 임계값 하향
        out = vis.visualize_graph('major_correlation_network.png')
        if out:
            print(f"Graph visualization saved as {out}")
    except Exception as e:
        print(f"Error: {e}")
