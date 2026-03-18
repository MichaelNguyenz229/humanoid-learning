# Speed Test — G1 Forward Velocity Limit

**Date:** 2026-03-18
**Policy:** Unitree G1 pretrained walking policy (motion.pt)
**Status:** PLANNED

## Question
What is the maximum commanded forward velocity (vx) the policy 
can sustain without falling?

## Method
- Terrain: flat
- Sweep vx from 0.2 → 2.0 m/s in 0.2 increments
- 5 episodes per speed
- Episode length: 10 seconds max
- Fall = torso height drops below 0.4m

## Success Criteria
Find the speed threshold where fall rate exceeds 40%

## Results
**
```

---

## CSV Schema
```
run_id, commanded_vx, episode, outcome, survival_time_s, fall_detected