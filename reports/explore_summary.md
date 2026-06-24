# 데이터 탐색 요약

- 본주행 파일: **116개**, 피실험자: **29명** [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29]
- 시나리오별 자차(uv) 행수: {1: 298214, 2: 335760, 3: 319114, 4: 362949}
- 총 자차 행수: **1,316,037** (샘플링 ~43.7Hz, dt≈0.0229s)
- drivingMode 분포: {'Manual': 1316037}

## 행동(Action) 후보 컬럼

| col | present | min | max | mean | nan | inf |
|---|---|---|---|---|---|---|
| steering | Y | -0.06745 | 0.05864 | -0.0001788 | 0 | 0 |
| appliedSteering | Y | -0.06745 | 0.05864 | -0.0001791 | 0 | 0 |
| rawSteering | Y | -0.06745 | 0.05864 | -0.0001788 | 0 | 0 |
| steeringVelocity | Y | -0.1615 | 0.1921 | 6.164e-05 | 0 | 0 |
| throttle | Y | 0 | 1 | 0.1799 | 0 | 0 |
| appliedThrottle | Y | 0 | 1 | 0.1799 | 0 | 0 |
| rawThrottle | Y | 0 | 1 | 0.1799 | 0 | 0 |
| brake | Y | 0 | 1 | 0.01283 | 0 | 0 |
| appliedBrake | Y | 0 | 1 | 0.01274 | 0 | 0 |
| rawBrake | Y | 0 | 1 | 0.01283 | 0 | 0 |

## 상태(State) 후보 컬럼

| col | present | min | max | mean | nan | inf |
|---|---|---|---|---|---|---|
| speedInKmPerHour | Y | -0.682 | 92.73 | 52.45 | 0 | 0 |
| speedInMetresPerSecond | Y | -0.1894 | 25.76 | 14.57 | 0 | 0 |
| localAccelInMetresPerSecond2 X | Y | -3.874 | 5.248 | 0.01329 | 0 | 0 |
| localAccelInMetresPerSecond2 Y | Y | -8.059 | 6.259 | 2.879e-05 | 0 | 0 |
| localAccelInMetresPerSecond2 Z | Y | -10.91 | 6.431 | 0.0621 | 0 | 0 |
| bodyRotSpeedInRadsPerSecond Yaw | Y | -0.1999 | 0.2608 | 0.0009424 | 0 | 0 |
| turningCurvature | Y | -6.463e+06 | 1.094e+07 | 24.66 | 0 | 0 |
| laneCurvature | Y | -0.01091 | 0.005138 | 2.331e-05 | 0 | 0 |
| offsetFromLaneCenter | Y | -1.441 | 1.232 | 0.1784 | 0 | 0 |
| offsetFromRoadCenter | Y | -6.832 | 13.49 | -1.922 | 0 | 0 |
| standardDeviationFromLaneCenter | Y | 0 | 0.2888 | 0.0164 | 0 | 0 |
| distanceToLeftBorder | Y | 3.16 | 6.283 | 4.995 | 0 | 0 |
| distanceToRightBorder | Y | 0.5174 | 3.089 | 1.494 | 0 | 0 |
| carriagewayWidth | Y | 6.187 | 6.802 | 6.489 | 0 | 0 |
| laneWidth | Y | 2.8 | 3.1 | 2.945 | 0 | 0 |
| distanceToFrontVehicle | Y | -0.1591 | 264.4 | 60.23 | 23915 | 0 |
| TTCToFrontVehicle | Y | -0.8262 | 3.268e+07 | 709.6 | 23915 | 640386 |
| roadLongitudinalSlope | Y | -0.03505 | 0.0493 | -0.00512 | 0 | 0 |
| roadLateralSlope | Y | -0.003925 | 0.0003212 | -9.411e-05 | 0 | 0 |
| speedLimit | Y | 50 | 50 | 50 | 0 | 0 |
| laneNumber | Y | 1 | 1 | 1 | 0 | 0 |
| leftLaneOverLap | Y | 0 | 0.5612 | 0.0008961 | 0 | 0 |
| rightLaneOverLap | Y | 0 | 0.3461 | 0.0118 | 0 | 0 |