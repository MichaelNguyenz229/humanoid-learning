"""
run_test.py — Humanoid simulation test runner

Usage:
    mjpython run_test.py --scene stairs1 --direction forward
    mjpython run_test.py --scene slope2  --direction backward
    mjpython run_test.py --scene octave3 --direction lateral --viewer
    mjpython run_test.py --list

Arguments:
    --scene      Name of scene XML in scenes/ folder (without .xml)
    --direction  forward | backward | lateral
    --viewer     Show interactive MuJoCo viewer (default: headless)
    --list       Print available scenes and exit
"""

import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# ── Available scenes ──────────────────────────────────────────────────────────
# Add entries here as you generate XMLs with the Unitree terrain tool
SCENES = {
    "flat": "Flat terrain — baseline",
    "stairs1":   "Stairs — low step height",
    "stairs2":   "Stairs — medium step height",
    "stairs3":  "Stairs — high step height",
    "slope1":    "Slope — shallow angle",
    "slope2":    "Slope — medium angle",
    "slope3":   "Slope — steep angle",
    "octave1":   "Uneven terrain — low octave (gentle bumps)",
    "octave2":   "Uneven terrain — mid octave",
    "octave3":  "Uneven terrain — high octave (rough surface)",
}

DIRECTIONS = ["forward", "backward", "lateral"]


def list_scenes():
    print("\nAvailable scenes:")
    for name, desc in SCENES.items():
        xml_path = os.path.join(os.path.dirname(__file__), "scenes", f"{name}.xml")
        status = "✅" if os.path.exists(xml_path) else "❌ xml missing"
        print(f"  {name:<15} {desc:<40} {status}")
    print(f"\nDirections: {', '.join(DIRECTIONS)}")
    print("\nExample:")
    print("  mjpython run_test.py --scene stairs1 --direction forward")
    print("  mjpython run_test.py --scene slope2 --direction backward --viewer\n")


def main():
    parser = argparse.ArgumentParser(
        description="Humanoid simulation test runner",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--scene",     "-s", type=str, help="Scene name (see --list)")
    parser.add_argument("--direction", "-d", type=str, choices=DIRECTIONS, help="Movement direction")
    parser.add_argument("--viewer",    "-v", action="store_true", help="Launch interactive viewer")
    parser.add_argument("--list",      "-l", action="store_true", help="List available scenes and exit")

    args = parser.parse_args()

    # ── List and exit ─────────────────────────────────────────────────────────
    if args.list:
        list_scenes()
        sys.exit(0)

    # ── Interactive fallback if args missing ──────────────────────────────────
    if args.scene is None:
        list_scenes()
        args.scene = input("Enter scene name: ").strip().lower()

    if args.direction is None:
        print(f"Directions: {', '.join(DIRECTIONS)}")
        args.direction = input("Enter direction: ").strip().lower()

    # ── Validate ──────────────────────────────────────────────────────────────
    if args.scene not in SCENES:
        print(f"Error: unknown scene '{args.scene}'. Run --list to see options.")
        sys.exit(1)

    scene_path = os.path.join(os.path.dirname(__file__), "scenes", f"{args.scene}.xml")
    if not os.path.exists(scene_path):
        print(f"Error: XML not found at {scene_path}")
        print("Generate it with the Unitree terrain tool first.")
        sys.exit(1)

    # ── Build output CSV path ─────────────────────────────────────────────────
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, f"{args.scene}_{args.direction}.csv")

    # ── Print run summary ─────────────────────────────────────────────────────
    print(f"\n── Run summary ───────────────────────────────────")
    print(f"   scene      {args.scene}  ({SCENES[args.scene]})")
    print(f"   direction  {args.direction}")
    print(f"   viewer     {'yes' if args.viewer else 'no (headless)'}")
    print(f"   output     {csv_path}")
    print(f"──────────────────────────────────────────────────\n")

    # ── Run ───────────────────────────────────────────────────────────────────
    from eval_speed_test import run
    run(
        scene_path=scene_path,
        direction=args.direction,
        csv_path=csv_path,
        viewer=args.viewer,
    )


if __name__ == "__main__":
    main()
