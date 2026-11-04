from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.spatial import cKDTree  # pyright: ignore[reportAttributeAccessIssue]

from . import Perception

if TYPE_CHECKING:
    from arena_humansim.core.agents import BaseAgent
    from arena_humansim.core.pool import AgentPool
    from arena_humansim.occlusion import Occluder
    from arena_humansim.utils.types import AgentState, BeliefState, WorldState

from arena_humansim.core.pool import KIND_ROBOT


class DefaultPerception(Perception):
    supports_pool: bool = True

    def __init__(self, occluder: Occluder | None = None, small_n_threshold: int = 64) -> None:
        self._occluder = occluder
        self._small_n_threshold = small_n_threshold
        self._shared_tree: cKDTree | None = None
        self._shared_ids: list[int] | None = None
        self._shared_positions: list[tuple[float, float]] | None = None

    def prepare_tick(self, all_agents: dict[int, AgentState]) -> None:
        ids: list[int] = []
        positions: list[tuple[float, float]] = []
        for aid, ag in all_agents.items():
            ids.append(aid)
            positions.append((ag.pose.x, ag.pose.y))
        self._shared_ids = ids
        self._shared_positions = positions
        if len(positions) > 1:
            self._shared_tree = cKDTree(np.array(positions, dtype=np.float64))
        else:
            self._shared_tree = None

    @property
    def shared_tree(self) -> cKDTree | None:
        return self._shared_tree

    @property
    def shared_ids(self) -> list[int] | None:
        return self._shared_ids

    @property
    def shared_positions(self) -> list[tuple[float, float]] | None:
        return self._shared_positions

    def compute_pool(self, pool: AgentPool) -> None:
        n = pool.n
        if n <= 1:
            pool.set_neighbor_csr(
                np.zeros(n + 1, dtype=np.int32),
                np.empty(0, dtype=np.int32),
            )
            return

        positions = pool.pos[:n]

        if n <= self._small_n_threshold:
            self._compute_pool_dense(pool, positions, n)
        else:
            self._compute_pool_kdtree(pool, positions, n)

    def _compute_pool_dense(self, pool: AgentPool, positions: np.ndarray, n: int) -> None:
        diff = positions[None, :, :] - positions[:, None, :]
        dists = np.hypot(diff[:, :, 0], diff[:, :, 1])
        np.fill_diagonal(dists, np.inf)

        vision_range = pool.vision_range[:n]
        mask = dists <= vision_range[:, None]

        proximity_sense = pool.proximity_sense[:n]
        prox_mask = dists <= proximity_sense[:, None] if np.any(proximity_sense > 0.0) else None

        vision_fov = pool.vision_fov[:n]
        needs_fov = vision_fov < 360.0
        if np.any(needs_fov):
            fov_pairs = mask & needs_fov[:, None]
            if prox_mask is not None:
                fov_pairs &= ~prox_mask
            r_fov, c_fov = np.nonzero(fov_pairs)
            if len(r_fov) > 0:
                heading = pool.theta[:n]
                cos_half = np.cos(np.radians(vision_fov * 0.5))
                dot = diff[r_fov, c_fov, 0] * np.cos(heading)[r_fov] + diff[r_fov, c_fov, 1] * np.sin(heading)[r_fov]
                outside = dot < dists[r_fov, c_fov] * cos_half[r_fov]
                mask[r_fov[outside], c_fov[outside]] = False

        if prox_mask is not None:
            mask |= prox_mask

        if self._occluder is not None:
            r_los, c_los = np.nonzero(mask & pool.vision_occlusion[:n][:, None])
            if len(r_los) > 0:
                blocked = ~_clear_unordered(self._occluder, positions, r_los, c_los, n)
                mask[r_los[blocked], c_los[blocked]] = False

        # Gate robot perception by observer's robot_awareness
        is_robot = pool.kind[:n] == KIND_ROBOT
        if np.any(is_robot):
            robot_cols = np.flatnonzero(is_robot)
            human_rows = np.flatnonzero(~is_robot)
            if len(human_rows) > 0 and len(robot_cols) > 0:
                h_idx = human_rows[:, None]
                r_idx = robot_cols[None, :]
                awareness = pool.robot_awareness[human_rows, None]
                blind_mask = awareness <= 0.0
                mask[h_idx, r_idx] &= ~blind_mask

                fov_gain = pool.robot_fov_gain[human_rows, None]
                partial = ((awareness > 0.0) & (awareness < 1.0)) | (fov_gain != 1.0)
                if np.any(partial):
                    p_rows, p_cols = np.nonzero(partial)
                    orig_r = human_rows[p_rows]
                    orig_c = robot_cols[p_cols]
                    pair_dists = dists[orig_r, orig_c]
                    eff_range = pool.vision_range[orig_r] * pool.robot_awareness[orig_r]
                    eff_prox = pool.proximity_sense[orig_r] * pool.robot_awareness[orig_r]
                    beyond = pair_dists > np.maximum(eff_range, eff_prox)
                    mask[orig_r[beyond], orig_c[beyond]] = False

                    surviving = ~beyond & (pair_dists > eff_prox)
                    if np.any(surviving):
                        sr_r = orig_r[surviving]
                        sr_c = orig_c[surviving]
                        eff_fov = pool.vision_fov[sr_r] * pool.robot_fov_gain[sr_r]
                        needs_fov_check = eff_fov < 360.0
                        cos_half = np.cos(np.radians(np.clip(eff_fov, 0.0, 360.0) * 0.5))
                        heading = pool.theta[sr_r]
                        dot = diff[sr_r, sr_c, 0] * np.cos(heading) + diff[sr_r, sr_c, 1] * np.sin(heading)
                        outside_fov = needs_fov_check & (dot < pair_dists[surviving] * cos_half)
                        mask[sr_r[outside_fov], sr_c[outside_fov]] = False

        row, col = np.where(mask)
        if len(row) == 0:
            pool.set_neighbor_csr(
                np.zeros(n + 1, dtype=np.int32),
                np.empty(0, dtype=np.int32),
            )
            return

        counts = mask.sum(axis=1)
        indptr = np.zeros(n + 1, dtype=np.int32)
        np.cumsum(counts, out=indptr[1:])
        col = col[np.lexsort((pool.agent_ids[col], row))]
        pool.set_neighbor_csr(indptr, col.astype(np.int32))

    def _compute_pool_kdtree(self, pool: AgentPool, positions: np.ndarray, n: int) -> None:
        tree = cKDTree(positions)
        self._shared_tree = tree

        max_range = float(max(pool.vision_range[:n].max(), pool.proximity_sense[:n].max()))
        sdm = tree.sparse_distance_matrix(tree, max_range, output_type="coo_matrix")

        row = sdm.row
        col = sdm.col
        dist = sdm.data

        mask = row != col
        row = row[mask]
        col = col[mask]
        dist = dist[mask]

        if len(row) == 0:
            pool.set_neighbor_csr(
                np.zeros(n + 1, dtype=np.int32),
                np.empty(0, dtype=np.int32),
            )
            return

        prox_ok = dist <= pool.proximity_sense[row]
        range_ok = dist <= pool.vision_range[row]
        keep = range_ok | prox_ok

        vision_fov = pool.vision_fov[:n]
        fov_idx = np.flatnonzero(range_ok & ~prox_ok & (vision_fov[row] < 360.0))
        if len(fov_idx) > 0:
            heading = pool.theta[:n]
            hx = np.cos(heading)
            hy = np.sin(heading)
            cos_half = np.cos(np.radians(vision_fov * 0.5))
            r_fov = row[fov_idx]
            c_fov = col[fov_idx]
            dx = positions[c_fov, 0] - positions[r_fov, 0]
            dy = positions[c_fov, 1] - positions[r_fov, 1]
            outside = dx * hx[r_fov] + dy * hy[r_fov] < dist[fov_idx] * cos_half[r_fov]
            keep[fov_idx[outside]] = False

        if self._occluder is not None:
            los_idx = np.flatnonzero(keep & pool.vision_occlusion[row])
            if len(los_idx) > 0:
                blocked = ~_clear_unordered(self._occluder, positions, row[los_idx], col[los_idx], n)
                keep[los_idx[blocked]] = False

        # Gate robot perception by observer's robot_awareness
        robot_nbr = (pool.kind[col] == KIND_ROBOT) & (pool.kind[row] != KIND_ROBOT)
        if np.any(robot_nbr):
            awareness = pool.robot_awareness[row[robot_nbr]]
            robot_nbr_idx = np.flatnonzero(robot_nbr)
            blind_idx = robot_nbr_idx[awareness <= 0.0]
            keep[blind_idx] = False

            fov_gain = pool.robot_fov_gain[row[robot_nbr]]
            partial_sub = ((awareness > 0.0) & (awareness < 1.0)) | (fov_gain != 1.0)
            if np.any(partial_sub):
                partial_idx = robot_nbr_idx[partial_sub]
                p_row = row[partial_idx]
                p_col = col[partial_idx]
                p_dist = dist[partial_idx]
                eff_range = pool.vision_range[p_row] * pool.robot_awareness[p_row]
                eff_prox = pool.proximity_sense[p_row] * pool.robot_awareness[p_row]
                beyond = p_dist > np.maximum(eff_range, eff_prox)
                keep[partial_idx[beyond]] = False

                surviving = ~beyond & (p_dist > eff_prox)
                if np.any(surviving):
                    s_idx = partial_idx[surviving]
                    s_row = p_row[surviving]
                    s_col = p_col[surviving]
                    eff_fov = pool.vision_fov[s_row] * pool.robot_fov_gain[s_row]
                    needs_fov_check = eff_fov < 360.0
                    cos_half = np.cos(np.radians(np.clip(eff_fov, 0.0, 360.0) * 0.5))
                    heading = pool.theta[s_row]
                    dx = positions[s_col, 0] - positions[s_row, 0]
                    dy = positions[s_col, 1] - positions[s_row, 1]
                    outside = needs_fov_check & (dx * np.cos(heading) + dy * np.sin(heading) < p_dist[surviving] * cos_half)
                    keep[s_idx[outside]] = False

        row = row[keep]
        col = col[keep]

        counts = np.bincount(row, minlength=n)
        indptr = np.zeros(n + 1, dtype=np.int32)
        np.cumsum(counts, out=indptr[1:])

        order = np.lexsort((pool.agent_ids[col], row))
        indices = col[order].astype(np.int32)

        pool.set_neighbor_csr(indptr, indices)

    def compute(
        self,
        agent: BaseAgent,
        all_agents: dict[int, AgentState],
        world_state: WorldState,
        belief: BeliefState,
    ) -> BeliefState:
        agent_id = agent.state.agent_id
        vision_range: float = agent.params.perception.vision_range
        vision_fov: float = agent.params.perception.vision_fov
        proximity_sense: float = agent.params.perception.proximity_sense
        vision_occlusion: bool = agent.params.perception.vision_occlusion

        if agent_id not in all_agents:
            return belief

        if self._shared_tree is None or self._shared_ids is None or self._shared_positions is None:
            return belief

        observer = all_agents[agent_id]
        ox, oy = observer.pose.x, observer.pose.y

        if len(all_agents) <= 1:
            return belief

        omnidirectional = vision_fov >= 360.0
        half_fov = np.radians(vision_fov * 0.5)
        heading = observer.pose.theta
        robot_awareness: float = getattr(agent.params.perception, "robot_awareness", 1.0)
        robot_fov_gain: float = getattr(agent.params.perception, "robot_fov_gain", robot_awareness)

        query_radius = max(vision_range, proximity_sense)
        neighbor_indices = self._shared_tree.query_ball_point([ox, oy], query_radius)

        candidates: list[int] = []
        for idx in neighbor_indices:
            nid = self._shared_ids[idx]
            if nid == agent_id:
                continue
            nbr_is_robot = getattr(all_agents[nid], "kind", 0) == KIND_ROBOT
            if nbr_is_robot and robot_awareness <= 0.0:
                continue
            eff_range = vision_range * robot_awareness if nbr_is_robot else vision_range
            eff_prox = proximity_sense * robot_awareness if nbr_is_robot else proximity_sense
            eff_fov = vision_fov * robot_fov_gain if nbr_is_robot else vision_fov

            nx, ny = self._shared_positions[idx]
            d = float(np.hypot(nx - ox, ny - oy))
            within_prox = d <= eff_prox
            if not within_prox:
                if d > eff_range:
                    continue
                if eff_fov < 360.0:
                    bearing = np.arctan2(ny - oy, nx - ox)
                    angle_diff = abs(_wrap_angle(bearing - heading))
                    if angle_diff > np.radians(eff_fov * 0.5):
                        continue
            candidates.append(idx)

        if self._occluder is not None and vision_occlusion and candidates:
            p_a = np.array([[ox, oy]] * len(candidates), dtype=np.float64)
            p_b = np.array([self._shared_positions[idx] for idx in candidates], dtype=np.float64)
            los = self._occluder.clear(p_a, p_b)
            candidates = [idx for idx, clear in zip(candidates, los, strict=True) if clear]

        for idx in candidates:
            nid = self._shared_ids[idx]
            belief.observed_agents.append(all_agents[nid])

        return belief


def _wrap_angle(angle: float) -> float:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def _clear_unordered(occluder: Occluder, positions: np.ndarray, row: np.ndarray, col: np.ndarray, n: int) -> np.ndarray:
    lo = np.minimum(row, col).astype(np.int64)
    hi = np.maximum(row, col).astype(np.int64)
    keys, inverse = np.unique(lo * n + hi, return_inverse=True)
    return occluder.clear(positions[keys // n], positions[keys % n])[inverse]
