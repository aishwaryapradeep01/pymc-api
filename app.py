import arviz as az
import pymc as pm
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
import os


app = Flask(__name__)
cors = CORS(app)

apikey = os.environ.get('APIKEY')

def load_trace():

    with pm.Model() as logistic_model:
        x1_data = pm.MutableData("x1", [0.380285])
        x2_data = pm.MutableData("x2", [1.49])
        x3_data = pm.MutableData("x3", [-0.46])
        x4_data = pm.MutableData("x4", [-0.49])

        y_data = pm.ConstantData("y", [0])

        alpha = pm.Normal("alpha", mu=0, sigma=10)
        beta1 = pm.Normal("beta1", mu=0, sigma=10)
        beta2 = pm.Normal("beta2", mu=0, sigma=10)
        beta3 = pm.Normal("beta3", mu=0, sigma=10)
        beta4 = pm.Normal("beta4", mu=0, sigma=10)

        p = pm.Deterministic("p", pm.math.sigmoid(alpha + beta1 * x1_data + beta2 * x2_data + beta3 * x3_data + beta4 * x4_data))
        obs = pm.Bernoulli("obs", p=p, observed=y_data, shape=x1_data.shape[0])
        
        idata =  az.from_netcdf('modelx_trace.nc')
        
    return idata, logistic_model

def predict_proba(m1, m2, m3, target, runtime):
    idata, logistic_model = load_trace()
    means = np.array([2.30346108e-02, 1.13274455e-02, 5.65137302e-04, 6.18577130e+06])
    deviations = np.array([1.47889563e-02, 1.03374226e-02, 2.70090807e-02, 3.42810488e+06])
    measurements = np.array([m1,m2,m3])
    deviation = np.abs(measurements - target).sum() / 3
    std_deviation = measurements.std()
    momentum = (measurements[-1] - measurements[-2]) + (measurements[-2] - measurements[-3])
    
    values = [200000, 500000, 1000000, 2500000, 5000000, 7500000, 10000000, 12000000, 15000000]
    
    next_values_index = next((i for i, val in enumerate(values) if val > runtime), None)
    next_values = [runtime] + values[next_values_index:next_values_index + 3] if next_values_index is not None else []
    print(next_values)
    results = {}
    for value in next_values:
        x = np.array([[deviation, std_deviation, momentum, value]])
        print(x)
        with logistic_model:
            pm.set_data({"x1": (x[:, 0]-means[0])/deviations[0], "x2": (x[:, 1]-means[1])/deviations[1], "x3": (x[:, 2]-means[2])/deviations[2], "x4": (x[:, 3]-means[3])/deviations[3]})
            post_idata = pm.sample_posterior_predictive(
                idata, var_names=["p"]
            )
            p_samples = post_idata.posterior_predictive["p"].values.flatten()
            p_mean = p_samples.mean()
            p_mean_percentage = round(p_mean * 100, 2)
            results[value] = p_mean_percentage
    return results

@app.route('/predict', methods=['POST'])
def predict_failure():

    headers = request.headers
    auth = headers.get("X-API-KEY")
    if auth == apikey:
        try:
            json_data = request.get_json()
            payload = json_data['payload']
            result = predict_proba(*payload)
                
            return {"pred_mean": result} , 200

        except Exception as e:
            return jsonify({'Error': str(e)}), 400
    else:
        return jsonify({"message": "ERROR: Unauthorized"}), 401



# predict_proba(1.58, 1.56, 1.59, 1.5, 5000000)

if __name__ == '__main__':
    app.run(host ='0.0.0.0', port = 8080, debug = False)
    