# Stair Height Experiment

## Setup
- Robot: Unitree G1
- Policy: motion.pt (pretrained walking)
- CMD velocity: 2.0 m/s forward
- Terrain: 5-step staircase, varied step height

## Conditions Tested
| step_height | description |
|-------------|-------------|
| 0.225m | 50% taller than baseline |
| 0.15m | baseline |
| 0.075m | 50% shorter than baseline |

## Results
| step_height | run 1 | run 2 | run 3 | avg fall time | avg fall X | steps reached |
|-------------|-------|-------|-------|---------------|------------|---------------|
| 0.225 | 9.7s | 8.8s | 8.9s | 9.1s | 1.35m | step 2 |
| 0.15 | 10.0s | 9.0s | 7.1s | 8.7s | 1.77m | step 2 |
| 0.075 | 7.4s | 7.0s | 7.5s | 7.3s | 2.18m | step 4 |

## Key Finding
[]

## Next Steps
[]