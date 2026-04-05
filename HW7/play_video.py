import numpy as np
import gymnasium as gym
import cv2
from DQN import DQNAgent


def record_checkpoints(
    checkpoint_paths,
    env_name="CartPole-v1",
    episodes_per_checkpoint=1,
    output_path="policy_progression.mp4",
    fps=30,
    font_scale=0.7,
    pause_frames=30,
):
    """
    Play each checkpoint in order and stitch all episodes into one video.

    Args:
        checkpoint_paths: list of paths to .pt checkpoint files (in desired order)
        env_name: Gymnasium environment name
        episodes_per_checkpoint: how many episodes to record per checkpoint
        output_path: where to save the .mp4
        fps: video frame rate
        font_scale: size of the overlay text
        pause_frames: number of blank-ish frames between checkpoints
    """
    env = gym.make(env_name, render_mode="rgb_array")
    all_frames = []

    for i, path in enumerate(checkpoint_paths):
        agent = DQNAgent.from_checkpoint(path, env=env)
        label = path.split("/")[-1].replace(".pt", "")

        for ep in range(episodes_per_checkpoint):
            state, _ = env.reset()
            terminated, truncated = False, False
            ep_reward = 0
            step = 0

            while not (terminated or truncated):
                frame = env.render()
                frame = annotate_frame(frame, label, ep_reward, step, font_scale)
                all_frames.append(frame)

                action_idx = agent.behavior_policy(state, mode="eval")
                action = agent.action_space_list[action_idx]
                state, reward, terminated, truncated, _ = env.step(action)
                ep_reward += reward
                step += 1

            # capture final frame with total reward
            frame = env.render()
            frame = annotate_frame(frame, label, ep_reward, step, font_scale)
            all_frames.append(frame)

            print(f"[{label}] Episode {ep + 1}: reward = {ep_reward:.0f}, steps = {step}")

        # add pause between checkpoints
        if i < len(checkpoint_paths) - 1 and len(all_frames) > 0:
            pause = np.copy(all_frames[-1])
            for _ in range(pause_frames):
                all_frames.append(pause)

    env.close()

    # write video
    h, w, _ = all_frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    for frame in all_frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    writer.release()
    print(f"\nSaved video to {output_path} ({len(all_frames)} frames, {len(all_frames)/fps:.1f}s)")


def annotate_frame(frame, label, reward, step, font_scale=0.7):
    """Burn checkpoint name, reward, and step count onto the frame."""
    frame = frame.copy()
    color = (255, 255, 255)
    bg = (0, 0, 0)
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 2

    lines = [
        f"Policy: {label}",
        f"Reward: {reward:.0f}  Step: {step}",
    ]

    for j, line in enumerate(lines):
        y = 25 + j * 30
        # black outline for readability
        cv2.putText(frame, line, (10, y), font, font_scale, bg, thickness + 2)
        cv2.putText(frame, line, (10, y), font, font_scale, color, thickness)

    return frame


if __name__ == "__main__":
    import argparse
    import glob

    parser = argparse.ArgumentParser()
    parser.add_argument("--glob", type=str, default=None, help="Glob pattern, e.g. 'models/cartpole_step*.pt'")
    parser.add_argument("--best", type=str, default=None, help="Path to best checkpoint, appended last")
    parser.add_argument("--env", type=str)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--output", type=str, default="policy_progression.mp4")
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()

    
    paths = sorted(
        glob.glob(args.glob),
        key=lambda p: int("".join(filter(str.isdigit, p.split("step")[-1]))),
    )
    
    paths.append(args.best)

    record_checkpoints(
        checkpoint_paths=paths,
        env_name=args.env,
        episodes_per_checkpoint=args.episodes,
        output_path=args.output,
        fps=args.fps,
    )
