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
        
        # Plotly Gapminder 데이터셋 로드
        self.df = pd.read_csv(self.data_path)
        return self.df

    def calculate_correlations(self):
        """수치형 지표 간의 상관관계 계산"""
        if self.df is None:
            self.load_data()
        
        # 수치형 컬럼 선택 (pop, lifeExp, gdpPercap)
        # 연도별로 변할 수 있으므로, 최신 연도(2007) 기준 혹은 전체 평균으로 분석
        latest_year = self.df[self.df['year'] == 2007]
        numeric_df = latest_year[['pop', 'lifeExp', 'gdpPercap']]
        
        # 컬럼명 변경 (가독성)
        numeric_df.columns = ['Population', 'Life Expectancy', 'GDP per Cap']
        
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
                if abs(weight) >= threshold:
                    self.graph.add_edge(nodes[i], nodes[j], weight=weight)
        
        return self.graph

    def visualize_graph(self, output_path='graph_output.png'):
        """그래프 시각화 및 로컬 저장"""
        if self.graph is None:
            self.build_graph()
            
        plt.figure(figsize=(10, 8))
        pos = nx.spring_layout(self.graph)
        
        # Edge 굵기를 가중치(상관계수)에 비례하게 설정
        edges = self.graph.edges(data=True)
        weights = [abs(d['weight']) * 5 for u, v, d in edges]
        edge_colors = ['red' if d['weight'] < 0 else 'blue' for u, v, d in edges]
        
        nx.draw_networkx_nodes(self.graph, pos, node_size=2000, node_color='lightblue')
        nx.draw_networkx_edges(self.graph, pos, width=weights, edge_color=edge_colors)
        nx.draw_networkx_labels(self.graph, pos, font_size=12, font_family='sans-serif')
        
        plt.title("Statistical Indicators Correlation Network (Gapminder 2007)")
        plt.axis('off')
        plt.savefig(output_path)
        plt.close()
        return output_path

if __name__ == "__main__":
    # 간단한 실행 테스트
    data_file = os.path.expanduser("~/repo/remote_work/stat_graph_vis/data/gapminder_combined.csv")
    vis = StatGraphVis(data_file)
    vis.load_data()
    vis.calculate_correlations()
    vis.build_graph()
    vis.visualize_graph('gapminder_network.png')
    print("Graph visualization saved as gapminder_network.png")
