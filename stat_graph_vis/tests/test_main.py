import pytest
import os
import pandas as pd
from main import StatGraphVis

@pytest.fixture
def vis_instance():
    data_path = os.path.expanduser("~/repo/remote_work/stat_graph_vis/data/gapminder_combined.csv")
    return StatGraphVis(data_path)

def test_load_data(vis_instance):
    df = vis_instance.load_data()
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert 'lifeExp' in df.columns

def test_calculate_correlations(vis_instance):
    corr = vis_instance.calculate_correlations()
    assert corr.shape == (3, 3)
    assert 'Life Expectancy' in corr.columns

def test_build_graph(vis_instance):
    graph = vis_instance.build_graph(threshold=0.1)
    assert len(graph.nodes) > 0
    # 상관관계가 있는 에지가 하나라도 있어야 함
    assert len(graph.edges) > 0

def test_visualize_graph(vis_instance):
    output = 'test_graph.png'
    path = vis_instance.visualize_graph(output)
    assert os.path.exists(path)
    if os.path.exists(path):
        os.remove(path)
