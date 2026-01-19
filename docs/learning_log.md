# Learning Log

## Day 1 - [Today's Date]

### What I learned:
- Virtual environments vs project folders (they're separate!)
- PATH environment variables and why conda needs initialization
- Project structure best practices
- Integrated terminal in VSCode

### What I built:
- First MuJoCo simulation (falling sphere)
- Proper project structure with docs, experiments, models, src

### Challenges:
- conda not working in Command Prompt initially
- Fixed with `conda init cmd.exe`

### Next steps:
- Load actual humanoid model
- Understand the physics simulation loop
- Make the humanoid stand up

### Questions for later:
- How does reinforcement learning work with MuJoCo?
- What's the difference between Unitree H1 and Optimus?

---

## Day 1 Continued - Pre-trained RL Model

### Major milestone: Witnessed learned locomotion! 🤖

**What I did:**
- Cloned Unitree's official `unitree_rl_gym` repository
- Found pre-trained G1 walking model (`motion.pt`)
- Successfully ran the model in MuJoCo simulation

**Key observation:**
The movement was slower and still somewhat robotic, but fundamentally different from PD control. While PD control felt "stiff" (fighting to hold a pose), the learned policy showed **coordinated intelligence**. Watching the joint controls fluctuate in real-time - dozens of values changing in unison - made walking (such a simple human task) reveal its true complexity.

**Technical insight:**
- Neural network controls 23+ joints simultaneously
- Each joint receiving different commands 50+ times/second
- No hardcoded rules - pure learned behavior from trial and error
- The "intelligence" comes from emergent patterns the network discovered

**Emotional reaction:**
SUPER cool to see actual intelligence vs. mechanical stiffness. This is the difference between classical robotics and modern AI-driven control.

### Next steps:
- Understand how the training works
- Train my own model from scratch
- Compare training curves and behavior

---

## Day 2 - Vision Analysis & Test Results

### Vision vs Proprioception Investigation

**Question:** Does the G1 policy use vision?

**Method:** 
- Checked observation inputs (47 values - all numerical)
- Searched scene.xml for cameras (none found)
- Analyzed policy architecture

**Conclusion:**
This is a **proprioception-only policy** using:
- Joint encoders (position, velocity)
- IMU (orientation, angular velocity)
- Previous actions
- Gait phase timing

**No vision input** - explains stair-climbing failure:
- Cannot anticipate terrain changes
- Only reacts after physical contact
- Would need either: (1) vision for prediction, or (2) training with stairs in data

### Test Results Summary

**Stairs Test (15cm steps):**
- **70%:** Trips on step 1, cannot recover
- **20%:** Avoids stairs entirely (steers away)
- **10%:** Bumps edge, stumbles, restabilizes

**Key Insights:**
- Policy optimized for flat ground (shuffle gait)
- Has robust balance recovery for small perturbations
- Cannot handle sustained elevation changes
- Shows some environmental awareness (avoidance behavior)

### Next Steps

**Immediate (Sessions 3-4):**
- Create varied scenarios: slopes, obstacles, flat control
- Test all with same policy
- Build failure mode taxonomy

**Soon (Sessions 5-6):**
- Add camera rendering to scenes
- Visualize robot's perspective
- Analyze vision vs proprioception gap

**Later (Optional):**
- Train improved policy with terrain randomization
- Explore vision-based policies (GR00T, etc.)

### Technical Understanding Gained

- ✅ File path management in MuJoCo/XML
- ✅ Working directory context for relative paths
- ✅ Policy input/output structure
- ✅ Observation preprocessing and scaling
- ✅ PD control layer below RL policy
- ✅ Difference between model (.xml) and policy (.pt) files


---

## Day 2 continued - Slope Testing & Documentation

### What I Built
- Created slope scenario (10° incline)
- Tested policy on gradual elevation changes
- Built comprehensive test results document

### Technical Challenges Solved
**Problem:** Slope test robot was frozen
- **Root cause:** Observation vector building was incomplete (file truncated)
- **Solution:** Completed the observation construction and policy inference code
- **Learning:** Always verify the full control loop is present!

**Problem:** Slope facing wrong direction
- **Root cause:** Euler angle sign (positive vs negative rotation)
- **Solution:** Changed `euler="0 0.174 0"` to `euler="0 -0.174 0"`
- **Learning:** Rotation directions matter! Negative = opposite direction

### Key Insights
**Slope vs Stairs:**
- Slopes: ~40% success (gradual is manageable!)
- Stairs: 0% success (discrete too hard)
- **Insight:** Policy handles gradual changes better than discrete ones

**Failure modes on slope:**
- Robot climbs halfway, then body angle increases too much
- Falls off side when tilted
- Shows proprioception can't compensate for sustained orientation changes

### Skills Gained
- ✅ Debugging complex simulation issues
- ✅ Understanding observation vector structure
- ✅ Creating varied test scenarios
- ✅ Writing professional technical documentation
- ✅ Organizing portfolio-ready reports

### Documentation Created
- `test_results.md` - Professional summary of all test scenarios
- Comparison table showing terrain difficulty hierarchy
- Recommendations for policy improvement

### What's Next
- Option A: Add camera rendering (see robot's perspective)
- Option B: Build automated test runner (metrics collection)
- Option C: Train improved policy with terrain randomization

**Total time today:** ~2-3 hours  
**Feeling:** Confident! Built real portfolio content.