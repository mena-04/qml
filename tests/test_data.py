from src.data import load_data, make_preprocessor

def test_data():
    X,y=load_data()
    assert X.shape==(569,30)
    assert set(y)=={0,1}
    assert make_preprocessor().fit_transform(X[:40]).shape==(40,4)
