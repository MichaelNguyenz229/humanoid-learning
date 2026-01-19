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