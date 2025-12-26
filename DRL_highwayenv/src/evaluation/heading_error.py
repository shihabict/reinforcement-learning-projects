import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import SAC
from highway_env import utils  # wrap_to_pi
from DRL_highwayenv.src.custom_LKA_env import LaneKeepingEnv


def compute_lateral_offset(env) -> float:
    """
    Lateral offset (meters) in the lane coordinate frame.
    0 means center of lane. Negative means left (depending on lane frame).
    """
    v = env.vehicle
    lane = v.lane
    _, lat = lane.local_coordinates(v.position)
    return float(lat)


def compute_heading_error(env) -> float:
    """
    Heading error (radians): wrap_to_pi(vehicle_heading - lane_heading_at_vehicle_position)
    """
    v = env.vehicle
    lane = v.lane
    s, _ = lane.local_coordinates(v.position)
    lane_heading = lane.heading_at(s)
    return float(utils.wrap_to_pi(v.heading - lane_heading))


def settling_time(signal, dt, eps, require_consecutive=10) -> float:
    """
    Settling time: the first time index k where |signal| stays within eps for
    the rest of the episode.

    require_consecutive: robustness option to avoid "false settling" due to noise.
    Example: require 10 consecutive steps within eps before declaring settled.
    """
    sig = np.abs(np.asarray(signal))
    within = sig <= eps
    n = len(within)

    if n == 0:
        return np.inf

    # robust settling: find earliest k where we have consecutive 'within' for
    # require_consecutive steps AND everything after k is also within
    for k in range(n):
        end_k = min(n, k + require_consecutive)
        if within[k:end_k].all() and within[k:].all():
            return k * dt

    return np.inf


def run_one_episode(model, render=True,
                    duration=20, policy_frequency=5,
                    start_longitudinal=40.0, start_lateral=-0.6,
                    lateral_eps=0.10,   # settling tolerance band in meters (e.g., ±0.10 m)
                    heading_eps_deg=3.0 # not used for settling here, but we compute heading error anyway
                    ):
    """
    Runs ONE episode, collects signals, computes metrics, and returns arrays + metrics.
    """

    env = LaneKeepingEnv(render_mode="human" if render else None)

    # Apply evaluation config, including initial lateral offset disturbance
    env.configure({
        "duration": duration,
        "policy_frequency": policy_frequency,

        # Key: start with lateral offset -0.6 m
        # This assumes your env uses position_in_lane = [longitudinal, lateral]
        "position_in_lane": [start_longitudinal, start_lateral],
    })

    dt = 1.0 / float(policy_frequency)

    obs, info = env.reset()
    done = False

    t_list = []
    lat_list = []
    head_list = []
    reward_list = []
    speed_list = []

    # run until termination/truncation
    while not done:
        action, _ = model.predict(obs, deterministic=True)

        # Gymnasium env returns 5 items
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        t_list.append(len(t_list) * dt)
        lat_list.append(compute_lateral_offset(env))
        head_list.append(compute_heading_error(env))
        speed_list.append(float(env.vehicle.speed))
        reward_list.append(float(reward))

    env.close()

    # Metrics
    Ts_lat = settling_time(lat_list, dt, eps=lateral_eps, require_consecutive=10)

    head_deg = np.degrees(np.asarray(head_list))
    metrics = {
        "episode_return": float(np.sum(reward_list)),
        "steps": int(len(lat_list)),
        "dt": float(dt),
        "lateral_abs_mean_m": float(np.mean(np.abs(lat_list))),
        "lateral_abs_max_m": float(np.max(np.abs(lat_list))),
        "heading_abs_mean_deg": float(np.mean(np.abs(head_deg))),
        "heading_abs_max_deg": float(np.max(np.abs(head_deg))),
        "heading_rms_deg": float(np.sqrt(np.mean(head_deg ** 2))),
        "settling_time_lat_s": float(Ts_lat) if np.isfinite(Ts_lat) else np.inf,
        "settled_lat": bool(np.isfinite(Ts_lat)),
    }

    return (np.asarray(t_list),
            np.asarray(lat_list),
            np.asarray(head_list),
            np.asarray(speed_list),
            np.asarray(reward_list),
            metrics)


def plot_episode(t, lat, head, speed, metrics, log_dir, lateral_eps=0.10, title_prefix="LKA Evaluation"):
    head_deg = np.degrees(head)

    # 1) Lateral offset plot
    plt.figure()
    plt.plot(t, lat)
    plt.axhline(+lateral_eps, linestyle="--")
    plt.axhline(-lateral_eps, linestyle="--")

    Ts = metrics["settling_time_lat_s"]
    if np.isfinite(Ts):
        plt.axvline(Ts, linestyle="--")
        plt.text(Ts, 0.0, f"  Ts={Ts:.2f}s", rotation=90, va="bottom")

    plt.xlabel("Time (s)")
    plt.ylabel("Lateral offset (m)")
    plt.title(f"{title_prefix}: Lateral Offset (settling band ±{lateral_eps} m)")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{log_dir}/lateral_offset.png")


    # 2) Heading error plot
    plt.figure()
    plt.plot(t, head_deg)
    plt.axhline(0, linestyle="--")
    plt.xlabel("Time (s)")
    plt.ylabel("Heading error (deg)")
    plt.title(f"{title_prefix}: Heading Error")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{log_dir}/heading_error.png")


    # PLot speed
    plt.figure()
    plt.plot(t, speed)
    plt.xlabel("Time (s)")
    plt.ylabel("Speed (m/s)")
    plt.title(f"{title_prefix}: Speed Profile")
    plt.grid(True)
    plt.savefig(f"{log_dir}/speed_profile.png")


if __name__ == "__main__":
    log_dir = '../logs_lka_50/'
    # Load your trained model
    model = SAC.load("../sac_lane_keeping_500000")

    # Run one rendered episode and plot signals
    t, lat, head, speed, rewards, metrics = run_one_episode(
        model,
        render=True,            # show simulation
        duration=20,
        policy_frequency=5,
        start_longitudinal=40.0,
        start_lateral=-0.6,     # your requested disturbance
        lateral_eps=0.10        # settling threshold (adjust if needed)
    )

    print("Metrics:", metrics)

    plot_episode(t, lat, head, speed, metrics,log_dir, lateral_eps=0.10, title_prefix="Lane Keeping (Start lat=-0.6m)")
