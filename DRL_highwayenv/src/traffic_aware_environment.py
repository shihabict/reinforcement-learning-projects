import numpy as np
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.road.road import Road, RoadNetwork
from highway_env.road.lane import StraightLane, LineType, AbstractLane
from highway_env import utils


class LaneKeepingEnv(AbstractEnv):

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({
            # Observation
            "observation": {
                "type": "Kinematics",
                "vehicles_count": 5,  # ego + 4 neighbors
                "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
                "absolute": False,
                "order": "sorted",
                "flatten": True,
            },

            # Action (continuous steering + acceleration)
            "action": {
                "type": "ContinuousAction",
                "lateral": True,
                "longitudinal": True,
                "steering_range": [-np.pi/6, np.pi/6],
                "dynamical": True,
                "acceleration_range": [-3.0, 2.0],
            },

            "desired_thw": 1.2,  # desired time headway [s]
            "thw_penalty_weight": 1.0,  # weight for headway penalty

            "ttc_threshold": 2.0,  # TTC threshold [s]
            "ttc_penalty_weight": 2.0,  # weight for TTC penalty (usually stronger than THW)

            # Fixed scenario setup (NO random traffic spawning)
            "scenario_fixed_5cars": True,
            "front_gaps": [25.0, 55.0],   # meters ahead of ego
            "rear_gaps":  [25.0, 55.0],   # meters behind ego
            "front_speed_range": [10.0, 20.0],  # random speed range (m/s)
            "rear_speed_range":  [10.0, 20.0],  # random speed range (m/s)

            "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
            "min_gap": 4.5,  # safety spacing so vehicles don't overlap at spawn

            # Timing
            "simulation_frequency": 50,
            "policy_frequency": 50,
            "duration": 30,

            # Road setup
            "lanes_count": 1,
            "speed_limit": 25,
            "road_length": 800,

            # Ego init
            "initial_speed": 15,
            "initial_lane_id": 0,
            "initial_longitudinal": 50,

            # Rewards
            "collision_reward": -10.0,
            "offroad_reward": -10.0,
            "lane_center_reward": 1.0,
            "heading_reward": 0.5,
            "smooth_steer_reward": 0.05,
            "thw_penalty_weight": 1.0,
            "ttc_penalty_weight": 2.0,
            "desired_thw": 1.2,
            "ttc_threshold": 2.0,

            "offroad_terminal": True,
        })
        return config

    def _reset(self) -> None:
        self._make_road()
        self._make_vehicles()
        self._prev_action = 0.0

    def _front_gap_thw_ttc(self, ego):
        """
        Returns (gap_m, thw_s, ttc_s) relative to the closest front vehicle in the same lane.
        If no front vehicle exists, returns (None, None, None).
        """
        front, _ = self.road.neighbour_vehicles(ego, lane_index=ego.lane_index)
        if front is None:
            return None, None, None

        # Straight road: x-axis is longitudinal
        gap = float(front.position[0] - ego.position[0] - front.LENGTH)
        gap = max(gap, 1e-3)

        # Time headway: gap / ego_speed
        thw = gap / max(float(ego.speed), 1e-3)

        # TTC only matters if ego is closing in (ego faster than front)
        rel_speed = float(ego.speed - front.speed)
        if rel_speed > 0:
            ttc = gap / rel_speed
        else:
            ttc = np.inf

        return gap, thw, ttc


    def _make_road(self) -> None:
        net = RoadNetwork()
        lane_width = AbstractLane.DEFAULT_WIDTH
        c, s = LineType.CONTINUOUS, LineType.STRIPED

        length = self.config["road_length"]
        lanes = self.config["lanes_count"]

        for lane_id in range(lanes):
            start = np.array([0, lane_id * lane_width])
            end = np.array([length, lane_id * lane_width])
            line_types = [c, s] if lane_id == 0 else [s, c]
            net.add_lane(
                "a", "b",
                StraightLane(start, end, line_types=line_types, speed_limit=self.config["speed_limit"])
            )

        self.road = Road(network=net, np_random=self.np_random, record_history=False)

    # -----------------------------
    # Fixed 5-car spawn logic
    # -----------------------------
    def _is_spawn_safe(self, new_vehicle) -> bool:
        min_gap = float(self.config.get("min_gap", 10.0))
        for v in self.road.vehicles:
            if np.linalg.norm(v.position - new_vehicle.position) < min_gap:
                return False
        return True

    def _spawn_vehicle_at(self, vehicle_type, lane_index, longitudinal, speed):
        veh = vehicle_type.make_on_lane(
            self.road,
            lane_index,
            longitudinal=float(longitudinal),
            speed=float(speed),
        )
        veh.randomize_behavior()
        if self._is_spawn_safe(veh):
            self.road.vehicles.append(veh)
            return veh
        return None

    def _spawn_fixed_around_ego(self, vehicle_type):
        ego = self.controlled_vehicles[0]
        lane_index = ego.lane_index
        ego_s = float(ego.position[0])

        front_gaps = self.config["front_gaps"]
        rear_gaps = self.config["rear_gaps"]

        fmin, fmax = self.config["front_speed_range"]
        rmin, rmax = self.config["rear_speed_range"]

        # 2 vehicles in front
        for gap in front_gaps:
            s_pos = ego_s + float(gap)
            if s_pos >= self.config["road_length"]:
                continue
            speed = float(self.np_random.uniform(fmin, fmax))
            self._spawn_vehicle_at(vehicle_type, lane_index, s_pos, speed)

        # 2 vehicles behind
        for gap in rear_gaps:
            s_pos = ego_s - float(gap)
            if s_pos <= 0:
                continue
            speed = float(self.np_random.uniform(rmin, rmax))
            self._spawn_vehicle_at(vehicle_type, lane_index, s_pos, speed)

    def _make_vehicles(self) -> None:
        self.controlled_vehicles = []

        # --- Create ego first ---
        ego_lane = self.road.network.get_lane(("a", "b", self.config["initial_lane_id"]))
        ego_s0 = float(self.config["initial_longitudinal"])
        ego_pos = ego_lane.position(ego_s0, 0)

        ego_vehicle = self.action_type.vehicle_class(
            road=self.road,
            position=ego_pos,
            speed=float(self.config["initial_speed"]),
            heading=ego_lane.heading_at(ego_s0),
        )

        self.road.vehicles.append(ego_vehicle)
        self.controlled_vehicles.append(ego_vehicle)

        # --- Create exactly 4 traffic cars (2 front, 2 rear) ---
        vehicle_type = utils.class_from_path(self.config["other_vehicles_type"])
        if self.config.get("scenario_fixed_5cars", True):
            self._spawn_fixed_around_ego(vehicle_type)

    # -----------------------------
    # Reward / termination as you had
    # -----------------------------
    def _reward(self, action) -> float:
        v = self.vehicle

        # Terminal penalties
        if v.crashed:
            return self.config["collision_reward"]
        if self.config["offroad_terminal"] and (not v.on_road):
            return self.config["offroad_reward"]

        # --- Lane centering reward ---
        lane = v.lane
        _, lateral = lane.local_coordinates(v.position)

        lane_width = lane.width_at(0) if hasattr(lane, "width_at") else AbstractLane.DEFAULT_WIDTH
        lat_err = abs(lateral) / (lane_width / 2 + 1e-6)
        lane_center_term = 1.0 - np.clip(lat_err, 0.0, 1.0)

        # --- Heading alignment reward ---
        lane_heading = lane.heading_at(lane.local_coordinates(v.position)[0])
        heading_err = utils.wrap_to_pi(v.heading - lane_heading)
        heading_term = 1.0 - np.clip(abs(heading_err) / (np.pi / 4), 0.0, 1.0)

        # --- Smoothness penalty ---
        steer = float(action[0]) if isinstance(action, (list, np.ndarray)) else float(action)
        smooth_term = -abs(steer - self._prev_action)
        self._prev_action = steer

        # --- TTC + Headway safety penalties ---
        thw_penalty = 0.0
        ttc_penalty = 0.0

        gap, thw, ttc = self._front_gap_thw_ttc(v)

        # Headway penalty: penalize if THW < desired_thw
        if thw is not None:
            desired_thw = float(self.config["desired_thw"])
            if thw < desired_thw:
                # normalized penalty in [-1, 0]
                thw_penalty = -(desired_thw - thw) / desired_thw

        # TTC penalty: penalize if TTC < ttc_threshold
        if ttc is not None and np.isfinite(ttc):
            ttc_threshold = float(self.config["ttc_threshold"])
            if ttc < ttc_threshold:
                # normalized penalty in [-1, 0]
                ttc_penalty = -(ttc_threshold - ttc) / ttc_threshold
        # =========================================================

        reward = (
                self.config["lane_center_reward"] * lane_center_term
                + self.config["heading_reward"] * heading_term
                + self.config["smooth_steer_reward"] * smooth_term
                + self.config["thw_penalty_weight"] * thw_penalty
                + self.config["ttc_penalty_weight"] * ttc_penalty
        )

        return float(reward)

    def _is_terminated(self) -> bool:
        v = self.vehicle
        if v.crashed:
            return True
        if self.config["offroad_terminal"] and (not v.on_road):
            return True
        return False

    def _is_truncated(self) -> bool:
        return self.time >= self.config["duration"]
