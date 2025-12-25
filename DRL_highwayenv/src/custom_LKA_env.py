import numpy as np
from typing import Dict, Optional, Tuple, Text

from highway_env.envs.common.abstract import AbstractEnv
from highway_env.road.road import Road, RoadNetwork
from highway_env.road.lane import StraightLane, LineType, AbstractLane
from highway_env.vehicle.controller import ControlledVehicle
from highway_env import utils

class LaneKeepingEnv(AbstractEnv):
    """
    Custom Lane Keeping Assistant environment.
    Goal: keep ego centered in lane, aligned with lane heading, avoid off-road/collision.
    """

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({
            # Observation (start simple)
            "observation": {
                "type": "Kinematics",
                "vehicles_count": 1,
                "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
                "absolute": True,
                "order": "sorted",
                "flatten": True,
            },

            # Action (continuous steering only)
            "action": {
                "type": "ContinuousAction",
                "lateral": True,
                "longitudinal": False,
                "steering_range": [-np.pi/6, np.pi/6],
                "dynamical": True,
            },

            # Timing
            "simulation_frequency": 15,
            "policy_frequency": 15,     # agent decides 5 times/sec (smoother steering)
            "duration": 20,            # seconds

            # Road setup
            "lanes_count": 2,
            "speed_limit": 25,         # m/s
            "road_length": 400,        # meters

            # Ego init
            "initial_speed": 15,       # m/s
            "initial_lane_id": 0,
            "initial_longitudinal": 50,

            # Rewards (weights)
            "collision_reward": -5.0,
            "offroad_reward": -5.0,
            "lane_center_reward": 1.0,
            "heading_reward": 0.5,
            "smooth_steer_reward": 0.05,  # penalize big steering

            "offroad_terminal": True,
        })
        return config

    def _reset(self) -> None:
        self._make_road()
        self._make_vehicles()

        # For smoothness penalty (optional)
        self._prev_action = 0.0

    def _make_road(self) -> None:
        net = RoadNetwork()
        lane_width = AbstractLane.DEFAULT_WIDTH
        n, c, s = LineType.NONE, LineType.CONTINUOUS, LineType.STRIPED

        length = self.config["road_length"]
        lanes = self.config["lanes_count"]

        # Straight multi-lane road: "a" -> "b"
        for lane_id in range(lanes):
            start = np.array([0, lane_id * lane_width])
            end   = np.array([length, lane_id * lane_width])
            line_types = [c, s] if lane_id == 0 else [s, c]  # simple boundary styling
            net.add_lane("a", "b", StraightLane(start, end, line_types=line_types,
                                                speed_limit=self.config["speed_limit"]))

        self.road = Road(network=net, np_random=self.np_random, record_history=False)

    def _make_vehicles(self) -> None:
        self.controlled_vehicles = []

        ego_lane = self.road.network.get_lane(("a", "b", self.config["initial_lane_id"]))
        ego_pos = ego_lane.position(self.config["initial_longitudinal"], 0)

        # ControlledVehicle supports continuous steering control when using ContinuousAction
        ego_vehicle = self.action_type.vehicle_class(
            road=self.road,
            position=ego_pos,
            speed=self.config["initial_speed"],
            heading=ego_lane.heading_at(self.config["initial_longitudinal"]),
        )
        self.road.vehicles.append(ego_vehicle)
        self.controlled_vehicles.append(ego_vehicle)

    def _reward(self, action) -> float:
        v = self.vehicle

        # Offroad/collision penalties
        if v.crashed:
            return self.config["collision_reward"]
        if self.config["offroad_terminal"] and (not v.on_road):
            return self.config["offroad_reward"]

        # --- Lane centering reward ---
        # local_coordinates returns (longitudinal, lateral) coords in lane frame
        lane = v.lane
        _, lateral = lane.local_coordinates(v.position)

        # Normalize lateral error by lane width (0 = center, ~1 = on boundary)
        lane_width = lane.width_at(0) if hasattr(lane, "width_at") else AbstractLane.DEFAULT_WIDTH
        lat_err = abs(lateral) / (lane_width / 2 + 1e-6)
        lane_center_term = 1.0 - np.clip(lat_err, 0.0, 1.0)

        # --- Heading alignment reward ---
        lane_heading = lane.heading_at(lane.local_coordinates(v.position)[0])
        heading_err = utils.wrap_to_pi(v.heading - lane_heading)
        heading_term = 1.0 - np.clip(abs(heading_err) / (np.pi/4), 0.0, 1.0)

        # --- Steering smoothness penalty (optional) ---
        # For ContinuousAction, action is often a vector like [steer] or [steer, accel]
        steer = float(action[0]) if isinstance(action, (list, np.ndarray)) else float(action)
        smooth_term = -abs(steer - self._prev_action)
        self._prev_action = steer

        reward = (
            self.config["lane_center_reward"] * lane_center_term
            + self.config["heading_reward"] * heading_term
            + self.config["smooth_steer_reward"] * smooth_term
        )

        return reward

    def _is_terminated(self) -> bool:
        v = self.vehicle
        if v.crashed:
            return True
        if self.config["offroad_terminal"] and (not v.on_road):
            return True
        return False

    def _is_truncated(self) -> bool:
        return self.time >= self.config["duration"]
