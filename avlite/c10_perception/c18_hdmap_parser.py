from __future__ import annotations

import logging
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial import KDTree

from avlite.c10_perception.c11_perception_model import HDMap

log = logging.getLogger(__name__)


def parse_opendrive(hdmap: HDMap) -> None:
    """Parse an OpenDRIVE file into *hdmap* (roads, lanes, indexes, graphs)."""
    if not hdmap.xodr_file_name:
        log.error("No OpenDRIVE file specified.")
        return

    hdmap._point_to_road = {}
    hdmap._point_to_drivable_lane = {}
    hdmap._road_kdtree = None
    hdmap._lane_kdtree_drivable = None
    hdmap._all_road_points = []
    hdmap._all_drivable_lane_points = []

    hdmap.root = _parse_hdmap_xml(hdmap)
    if hdmap._all_road_points:
        hdmap._road_kdtree = KDTree(hdmap._all_road_points)
    if hdmap._all_drivable_lane_points:
        hdmap._lane_kdtree_drivable = KDTree(hdmap._all_drivable_lane_points)
    _connect_roads(hdmap)
    _connect_lanes(hdmap)

    log.debug(
        "Number of roads in HD Map: %d vs nodes in graph %d",
        len(hdmap.roads),
        len(hdmap.road_network.nodes()),
    )
    log.debug(
        "Number of lanes in HD Map: %d vs nodes in graph %d",
        len(hdmap.lanes),
        len(hdmap.lane_network.nodes()),
    )


def _parse_hdmap_xml(hdmap: HDMap) -> ET.Element | None:
    try:
        tree = ET.parse(hdmap.xodr_file_name)
        root = tree.getroot()
    except ET.ParseError as e:
        log.error("Error parsing OpenDRIVE file: %s", e)
        return None

    for road_element in root.findall("road"):
        plan_view = road_element.find("planView")
        if plan_view is None:
            continue

        road_x, road_y = [], []
        for geometry in plan_view.findall("geometry"):
            x0 = float(geometry.get("x", "0"))
            y0 = float(geometry.get("y", "0"))
            hdg = float(geometry.get("hdg", "0"))
            length = float(geometry.get("length", "0"))
            gtype = "line"
            attrib = {}
            for child in geometry:
                if child.tag in ["line", "arc", "spiral", "poly3", "paramPoly3"]:
                    gtype = child.tag
                    attrib = child.attrib
                    break

            x_vals, y_vals = sample_OpenDrive_geometry(
                x0, y0, hdg, length, gtype, attrib,
                n_pts=int(length // hdmap.sampling_resolution + 1),
            )
            road_x.extend(x_vals)
            road_y.extend(y_vals)

        p_id, s_id = "-1", "-1"
        predecessor = road_element.find("link/predecessor")
        if predecessor is not None:
            if predecessor.get("elementType") == "road":
                p_id = predecessor.get("elementId", "")
            elif predecessor.get("elementType") == "junction":
                p_id = predecessor.get("elementId", "")

        successor = road_element.find("link/successor")
        if successor is not None:
            if successor.get("elementType") == "road":
                s_id = successor.get("elementId", "")
            elif successor.get("elementType") == "junction":
                s_id = successor.get("elementId", "")

        it = 1
        while len(road_x) > 1 and road_x[-1] == road_x[-2] and road_y[-1] == road_y[-2]:
            log.warning(
                "%d x Road ID: %s - last two points are the same, removing (%s, %s)",
                it, road_element.get("id"), road_x[-1], road_y[-1],
            )
            road_y.pop()
            road_x.pop()
            it += 1
        while len(road_x) > 1 and road_x[0] == road_x[1] and road_y[0] == road_y[1]:
            log.warning(
                "%d x Road ID: %s - first two points are the same, removing (%s, %s)",
                it, road_element.get("id"), road_x[0], road_y[0],
            )
            road_y.pop(0)
            road_x.pop(0)
            it += 1

        r = HDMap.Road(
            id=road_element.get("id", ""),
            road_element=road_element,
            center_line=np.array([road_x, road_y]),
            pred_id=p_id,
            succ_id=s_id,
            length=float(road_element.get("length", "")),
            pred_type=predecessor.get("elementType", "") if predecessor is not None else "",
            succ_type=successor.get("elementType", "") if successor is not None else "",
            junction_id=road_element.get("junction", ""),
        )

        for x, y in zip(road_x, road_y):
            hdmap._point_to_road[(x, y)] = r
            hdmap._all_road_points.append((x, y))

        if hdmap.road_by_id.get(r.id) is not None:
            log.error("Adding Road ID %s, but it already exists in road_ids.", r.id)
        hdmap.road_by_id[r.id] = r
        hdmap.roads.append(r)
        _process_lane_sections(hdmap, r, road_x, road_y)

    return root


def _process_lane_sections(
    hdmap: HDMap, r: HDMap.Road, road_x: list[float], road_y: list[float]
) -> None:
    road = r.road_element
    lanes_sections = road.findall("lanes/laneSection")
    if not lanes_sections:
        return

    lane_offsets = road.findall("lanes/laneOffset")

    for i, lane_section in enumerate(lanes_sections):
        r.lane_sections.append([])
        s_section = float(lane_section.get("s", "0.0"))
        r.lane_section_s_vals.append(s_section)
        offset = _get_lane_offset_at_s(lane_offsets, s_section)

        for side in ["left", "right"]:
            lanes = lane_section.findall(f"{side}/lane")
            if side == "left":
                lanes.sort(key=lambda lane_el: int(lane_el.get("id", "0")))
            else:
                lanes.sort(key=lambda lane_el: int(lane_el.get("id", "0")), reverse=True)

            cumulative_offset = offset
            for lane_element in lanes:
                lane_id = int(lane_element.get("id", "0"))
                width_element = lane_element.find("width")

                if width_element is not None and lane_id != 0:
                    width = float(width_element.get("a", "0.0"))
                    if float(width_element.get("b")) != 0.0 and lane_element.get("type") == "driving":
                        log.warning(
                            "Lanes with variable width are not supported yet. "
                            "We assume fixed width of %.2f Lane ID: %d, type: %s, Road ID: %s",
                            width, lane_id, lane_element.get("type"), r.id,
                        )

                    cumulative_offset += width / 2 if side == "left" else -width / 2

                    pred_id = (
                        lane_element.find("link/predecessor").get("id", "")
                        if lane_element.find("link/predecessor") is not None
                        else ""
                    )
                    succ_id = (
                        lane_element.find("link/successor").get("id", "")
                        if lane_element.find("link/successor") is not None
                        else ""
                    )
                    lane = HDMap.Lane(
                        id=int(lane_element.get("id", "")),
                        uid=f"{r.id}_{lane_element.get('id', '')}",
                        lane_element=lane_element,
                        type=lane_element.get("type", ""),
                        pred_id=pred_id,
                        succ_id=succ_id,
                        road_id=r.id,
                        lane_section_idx=i,
                        road=r,
                        width=width,
                    )
                    lane.side = "right" if int(lane.id) < 0 else "left"
                    hdmap.lanes.append(lane)
                    r.lane_sections[i].append(lane)

                    lane_x, lane_y = _get_lane_xy_path(road_x, road_y, cumulative_offset)
                    if lane_x and lane_y:
                        lane.center_line = np.array([lane_x, lane_y])
                        for x, y in zip(lane_x, lane_y):
                            x = float(x)
                            y = float(y)
                            if lane.type == "driving":
                                hdmap._point_to_drivable_lane[(x, y)] = lane
                                hdmap._all_drivable_lane_points.append((x, y))

                    cumulative_offset += width / 2 if side == "left" else -width / 2


def _get_lane_xy_path(
    road_x: list[float], road_y: list[float], offset: float
) -> tuple[list[float], list[float]] | tuple[None, None]:
    assert len(road_x) == len(road_y), "Road X and Y coordinates must be of the same length."

    if not road_x or len(road_x) < 2:
        return None, None

    lane_x, lane_y = [], []

    for i in range(len(road_x)):
        try:
            prev_idx = i - 1
            next_idx = i + 1

            if prev_idx >= 0 and next_idx < len(road_x):
                dx1 = road_x[i] - road_x[prev_idx]
                dy1 = road_y[i] - road_y[prev_idx]
                dx2 = road_x[next_idx] - road_x[i]
                dy2 = road_y[next_idx] - road_y[i]
                dx = (dx1 + dx2) / 2
                dy = (dy1 + dy2) / 2
            elif prev_idx >= 0:
                if prev_idx > 0:
                    dx1 = road_x[prev_idx] - road_x[prev_idx - 1]
                    dy1 = road_y[prev_idx] - road_y[prev_idx - 1]
                    dx2 = road_x[i] - road_x[prev_idx]
                    dy2 = road_y[i] - road_y[prev_idx]
                    dx = (dx1 + dx2) / 2
                    dy = (dy1 + dy2) / 2
                else:
                    dx = road_x[i] - road_x[prev_idx]
                    dy = road_y[i] - road_y[prev_idx]
            elif next_idx < len(road_x):
                if next_idx + 1 < len(road_x):
                    dx1 = road_x[next_idx] - road_x[i]
                    dy1 = road_y[next_idx] - road_y[i]
                    dx2 = road_x[next_idx + 1] - road_x[next_idx]
                    dy2 = road_y[next_idx + 1] - road_y[next_idx]
                    dx = (dx1 + dx2) / 2
                    dy = (dy1 + dy2) / 2
                else:
                    dx = road_x[next_idx] - road_x[i]
                    dy = road_y[next_idx] - road_y[i]
            else:
                continue

            length = np.sqrt(dx * dx + dy * dy)
            if length > 0:
                nx = -dy / length
                ny = dx / length
                lane_x.append(road_x[i] + nx * offset)
                lane_y.append(road_y[i] + ny * offset)

        except (IndexError, ValueError) as e:
            log.error("Error processing lane boundary: %s", e)
            continue

    return lane_x, lane_y


def _get_lane_offset_at_s(lane_offsets, s) -> float:
    if not lane_offsets:
        return 0.0

    applicable_offset = None
    for offset in lane_offsets:
        offset_s = float(offset.get("s", "0.0"))
        if offset_s <= s:
            applicable_offset = offset
        else:
            break

    if applicable_offset is None:
        return 0.0

    offset_s = float(applicable_offset.get("s", "0.0"))
    local_s = s - offset_s
    a = float(applicable_offset.get("a", "0.0"))
    b = float(applicable_offset.get("b", "0.0"))
    c = float(applicable_offset.get("c", "0.0"))
    d = float(applicable_offset.get("d", "0.0"))

    return a + b * local_s + c * local_s ** 2 + d * local_s ** 3


def _get_connecting_roads_from_junction(root, road_element, junction_id):
    junction = root.find(f".//junction[@id='{junction_id}']")
    if junction is None:
        log.error(
            "Junction with ID %s not found. road_id: %s",
            junction_id, road_element.get("id"),
        )
        return []

    successor_roads = []
    road_id = road_element.get("id", "")

    for connection in junction.findall("connection"):
        incoming_road = connection.get("incomingRoad")
        if incoming_road == road_id:
            connecting_road = connection.get("connectingRoad")
            if connecting_road:
                successor_roads.append(connecting_road)

    return successor_roads


def _connect_lanes(hdmap: HDMap) -> None:
    lane_by_uid = {f"{lane.road_id}_{lane.id}": lane for lane in hdmap.lanes if lane.type == "driving"}
    hdmap.lane_by_uid = lane_by_uid

    for lane in hdmap.lanes:
        if lane.type != "driving":
            continue

        if lane.pred_id and lane.pred_type == "lane":
            for pred_road in lane.road.predecessors:
                pred_uid = f"{pred_road.id}_{lane.pred_id}"
                pred_lane = lane_by_uid.get(pred_uid)
                lane.neighbors.add(pred_lane)
                pred_lane.neighbors.add(lane)

        if lane.succ_id and lane.succ_type == "lane":
            for succ_road in lane.road.successors:
                succ_uid = f"{succ_road.id}_{lane.succ_id}"
                succ_lane = lane_by_uid.get(succ_uid)
                lane.neighbors.add(succ_lane)
                succ_lane.neighbors.add(lane)

    for la in hdmap.lanes:
        for lb in la.neighbors:
            if hdmap.can_laneA_access_laneB(la, lb):
                la.drivable_neighbors.add(lb)
                hdmap.lane_network.add_edge(la.uid, lb.uid, weight=la.road.length, lane_change=False)

    for road in hdmap.roads:
        for lane_section in road.lane_sections:
            for lane in lane_section:
                if lane.type == "driving":
                    for other_lane in lane_section:
                        if (
                            other_lane.type == "driving"
                            and lane != other_lane
                            and int(lane.id) * int(other_lane.id) > 0
                        ):
                            hdmap.lane_network.add_edge(lane.uid, other_lane.uid, weight=0.0, lane_change=True)
                            hdmap.lane_network.add_edge(other_lane.uid, lane.uid, weight=0.0, lane_change=True)


def _connect_roads(hdmap: HDMap) -> None:
    for r in hdmap.roads:
        if r.pred_id != "" and r.pred_type == "road":
            pred_road = hdmap.road_by_id.get(r.pred_id, None)
            if pred_road:
                r.predecessors.append(pred_road)

        if r.succ_id != "" and r.succ_type == "road":
            succ_road = hdmap.road_by_id.get(r.succ_id, None)
            if succ_road:
                r.successors.append(succ_road)
                hdmap.road_network.add_edge(r.id, r.succ_id, weight=r.length)

        if hdmap.road_has_driving_lanes(r):
            p_ids, s_ids = [], []
            successor = r.road_element.find("link/successor")
            if successor is not None and successor.get("elementType") == "junction":
                junction_id = successor.get("elementId", "")
                s_ids = _get_connecting_roads_from_junction(hdmap.root, r.road_element, junction_id)

            predecessor = r.road_element.find("link/predecessor")
            if predecessor is not None and predecessor.get("elementType") == "junction":
                junction_id = predecessor.get("elementId", "")
                p_ids = _get_connecting_roads_from_junction(hdmap.root, r.road_element, junction_id)

            for succ in s_ids:
                s_road = hdmap.road_by_id.get(succ, None)
                assert s_road is not None, f"Road ID {succ} not found in road_ids."
                if hdmap.road_has_driving_lanes(s_road):
                    hdmap.road_network.add_edge(r.id, succ, weight=r.length)
                    r.successors.append(hdmap.road_by_id[succ])

            for pred in p_ids:
                p_road = hdmap.road_by_id.get(pred, None)
                assert p_road is not None, f"Road ID {pred} not found in road_ids."
                if hdmap.road_has_driving_lanes(p_road):
                    hdmap.road_network.add_edge(pred, r.id, weight=r.length)
                    r.predecessors.append(hdmap.road_by_id[pred])


def sample_OpenDrive_geometry(x0, y0, hdg, length, geom_type="line", attributes=None, n_pts=50):
    """Return (x_vals, y_vals) for OpenDRIVE geometry segments."""
    x_vals, y_vals = [], []
    s_array = np.linspace(0, length, n_pts)

    if len(s_array) > 1:
        if float(s_array[-1]) == float(s_array[-2]):
            log.warning("Length is too small to sample points for %s at %s, %s?.", geom_type, x0, y0)
            log.warning("last two points: %s, %s", s_array[-1], s_array[-2])

    if geom_type == "arc" and attributes is not None:
        curvature = float(attributes.get("curvature", 0))
        if curvature != 0:
            radius = abs(1.0 / curvature)
            arc_direction = np.sign(curvature)
            center_x = x0 - np.sin(hdg) * radius * arc_direction
            center_y = y0 + np.cos(hdg) * radius * arc_direction
            start_angle = np.arctan2(y0 - center_y, x0 - center_x)
            dtheta = length / radius * arc_direction
            angles = np.linspace(start_angle, start_angle + dtheta, n_pts)
            for angle in angles:
                x_vals.append(center_x + radius * np.cos(angle))
                y_vals.append(center_y + radius * np.sin(angle))
            if len(x_vals) > 1:
                assert x_vals[-1] != x_vals[-2], f"last two points: {x_vals[-1]}, {x_vals[-2]}"
            return x_vals, y_vals

    elif geom_type == "spiral" and attributes is not None:
        curvStart = float(attributes.get("curvStart", 0))
        curvEnd = float(attributes.get("curvEnd", 0))

        if abs(curvStart) < 1e-10 and abs(curvEnd) < 1e-10:
            for s in s_array:
                x_vals.append(x0 + s * np.cos(hdg))
                y_vals.append(y0 + s * np.sin(hdg))
            if len(x_vals) > 1:
                assert x_vals[-1] != x_vals[-2], f"last two points: {x_vals[-1]}, {x_vals[-2]}"
            return x_vals, y_vals

        current_x, current_y = x0, y0
        current_hdg = hdg

        for i in range(n_pts - 1):
            s_start = s_array[i]
            s_end = s_array[i + 1]
            s_mid = (s_start + s_end) / 2
            segment_length = s_end - s_start
            t = s_mid / length
            current_curv = curvStart + t * (curvEnd - curvStart)

            if abs(current_curv) < 1e-10:
                next_x = current_x + segment_length * np.cos(current_hdg)
                next_y = current_y + segment_length * np.sin(current_hdg)
            else:
                radius = abs(1.0 / current_curv)
                arc_direction = np.sign(current_curv)
                dtheta = segment_length / radius * arc_direction
                next_hdg = current_hdg + dtheta
                next_x = current_x + segment_length * np.cos((current_hdg + next_hdg) / 2)
                next_y = current_y + segment_length * np.sin((current_hdg + next_hdg) / 2)
                current_hdg = next_hdg

            x_vals.append(current_x)
            y_vals.append(current_y)
            current_x, current_y = next_x, next_y

        x_vals.append(current_x)
        y_vals.append(current_y)
        if len(x_vals) > 1:
            assert x_vals[-1] != x_vals[-2], f"last two points: {x_vals[-1]}, {x_vals[-2]}"
        return x_vals, y_vals

    elif geom_type in ["poly3", "paramPoly3"] and attributes is not None:
        log.error("Unsupported geometry type: %s. Will use default line approximation.", geom_type)

    for s in s_array:
        x_vals.append(x0 + s * np.cos(hdg))
        y_vals.append(y0 + s * np.sin(hdg))

    if len(x_vals) > 1:
        assert x_vals[-1] != x_vals[-2], f"last two points: {x_vals[-1]}, {x_vals[-2]}"
    return x_vals, y_vals
