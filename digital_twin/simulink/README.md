# Simulink / MATLAB integration

Suggested subsystems: sensor input -> preprocessing -> feature engineering -> physics engine -> AI prediction ->
state estimation -> health estimation -> degradation -> decision logic.

The AI block calls the Python service over REST, so MATLAB never needs the Python ML stack:

```matlab
% inside a MATLAB Function block / script (simulation-only: "sensor" data comes from the datasets)
opts = weboptions('MediaType','application/json','Timeout',30);
body = struct('bearing_id','B001','fs',12000,'rpm',1800,'vibration',vib(:)', ...
              'current1',i1(:)','current2',i2(:)','temperature',45);
res  = webwrite('http://localhost:8000/predict', body, opts);
fault = res.fault;  health = res.health;  alert = res.alert.level;
```

Start the service with `uvicorn backend.main:app --port 8000`. The Python twin in `state_estimator.py`
is the reference implementation; Simulink can mirror it (Kalman filter on health, trend, alert rules).
