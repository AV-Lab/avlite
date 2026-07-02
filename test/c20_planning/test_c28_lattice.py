"""Unit tests for lattice planning primitives (avlite.c20_planning.c28_lattice).

Tests verify:
- Node identity hashing is stable for equal state.
- Edge initialization builds a local quintic trajectory.
- Lattice.reset() clears accumulated nodes and edges.
"""

from avlite.c20_planning.c28_lattice import Edge, Lattice, Node
from avlite.c60_common.c63_trajectory_tracker import TrajectoryTracker


def _straight_global(x_end: float = 100.0, n: int = 20) -> TrajectoryTracker:
    xs = [x_end * i / (n - 1) for i in range(n)]
    path = [(x, 0.0) for x in xs]
    left = [3.0] * n
    right = [-3.0] * n
    tj = TrajectoryTracker(path=path, velocity=[5.0] * n)
    tj.ref_left_boundary_d = left
    tj.ref_right_boundary_d = right
    return tj


class TestLatticeNode:
    def test_equal_nodes_share_hash(self):
        a = Node(s=1.0, d=0.0, x=1.0, y=0.0)
        b = Node(s=1.0, d=0.0, x=1.0, y=0.0)
        assert hash(a) == hash(b)


class TestLatticeEdge:
    def test_edge_builds_local_trajectory(self):
        global_tj = _straight_global()
        start = Node(s=0.0, d=0.0, x=0.0, y=0.0)
        end = Node(s=20.0, d=0.0, x=20.0, y=0.0)
        edge = Edge(start=start, end=end, global_tj=global_tj, num_of_points=10)
        assert edge.local_trajectory is not None
        assert len(edge.local_trajectory.path_x) == 10


class TestLatticeReset:
    def test_reset_clears_accumulated_state(self):
        global_tj = _straight_global()
        lattice = Lattice(
            global_trajectory=global_tj,
            ref_left_boundary_d=[3.0] * len(global_tj.path),
            ref_right_boundary_d=[-3.0] * len(global_tj.path),
            planning_horizon=1,
        )
        lattice.nodes.append(Node(s=5.0, d=0.0, x=5.0, y=0.0))
        lattice.lattice_nodes_by_level[0].append(Node(s=0.0, d=0.0, x=0.0, y=0.0))
        lattice.reset()
        assert lattice.nodes == []
        assert lattice.edges == []
        assert len(lattice.lattice_nodes_by_level) == 0
