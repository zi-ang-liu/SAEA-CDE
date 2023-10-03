from sklearn.ensemble import RandomForestRegressor

def build_model(x, y):

    model = RandomForestRegressor()
    model.fit(x, y)
    
    return model