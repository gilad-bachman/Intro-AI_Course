# AI AND SOURCE DISCLOSURE
# Codex and Claude helped tackle the problem through research, deriving ideas
# from papers, and writing, debugging and testing code.
# Also they helped me undertand which ideas works and which do not, and how to implement them.
#
# Sources used for related problems and heuristic/pruning ideas:
# - TransportEditor (Skopek and Bartak, ICAPS 2017):
#   https://oskopek.com/docs/ICAPS17-TransportEditor-Demo.pdf
# - CSU Planning Repository, Ferry/Logistics domains:
#   https://www.cs.colostate.edu/meps/repository/ucpop.html
# - IPC 2002 domains, especially numeric Depots:
#   https://ipc02.icaps-conference.org/domains.html
# - CTU planning exercise (search-result description only):
#   https://cw.fel.cvut.cz/old/courses/ae3b33kui/seminars_and_labs/11
# - AI-Planning PDDL benchmark generators:
#   https://github.com/AI-Planning/pddl-generators
# - Planning.Domains benchmark collections:
#   https://api.planning.domains/
# - IBM ACPBench, action applicability and plan validation:
#   https://ibm.github.io/ACPBench/
# - Efficient Stubborn Sets (Wehrle and Helmert, ICAPS 2014):
#   https://ai.dmi.unibas.ch/papers/wehrle-helmert-icaps2014.pdf
# - Integrating Partial Order Reduction and Symmetry Elimination
#   (Wehrle et al., IJCAI 2015):
#   https://ijcai.org/proceedings/2015/0244.pdf
# - Getting the Most Out of Pattern Databases (Pommerening et al.,
#   IJCAI 2013; proceedings abstract consulted):
#   https://ijcai.org/proceedings/2013/347
# - References inherited from the adapted branch (not read directly here):
#   Paul et al., SOCS 2017:
#   https://ojs.aaai.org/index.php/SOCS/article/download/18428/18219/21944
#   Additive Pattern Database Heuristics (Felner, Korf and Hanan):
#   https://arxiv.org/abs/1107.0050

from collections import OrderedDict
import itertools
import math
import time

try:
    import numpy as np
except ImportError:
    np = None  # Cargo tables are optional; the ordinary bounds still work.

import search

id = ["211397054"]


class _CargoPattern:
    """Finite cargo abstraction, solved by monotone Bellman lower bounds.

    Empty vehicles are available everywhere and all use the largest capacity.
    A state retains ground parcels and the cargo groups of nonempty trucks.
    Real actions either project to an abstract action of no greater assigned
    cost, or disappear at zero cost. No real logistics states are searched.
    """
    @classmethod
    def build(cls, weights, goals, places, distances, capacity, deadline):
        if (np is None or not 1 <= len(weights) <= 6 or capacity <= 0 or max(weights) > capacity or
                time.perf_counter() >= deadline):
            return None
        signatures = sorted(set(zip(weights, goals)))
        kinds = {signature: i for i, signature in enumerate(signatures)}
        labels = tuple(kinds[x] for x in zip(weights, goals))
        count = len(signatures)
        full = (1 << len(weights)) - 1
        subsets = {}
        for mask in range(1, full + 1):
            cargo = tuple(sorted(labels[p] for p in range(len(weights)) if mask & (1 << p)))
            if sum(signatures[c][0] for c in cargo) <= capacity:
                subsets[mask] = cargo
        cargo_types = sorted(set(subsets.values()), key=lambda x: (len(x), x))
        cargo_ids = {cargo: count + i for i, cargo in enumerate(cargo_types)}
        partitions = {0: {()}}
        for mask in range(1, full + 1):
            first = mask & -mask
            p = first.bit_length() - 1
            choices = {tuple(sorted((labels[p],) + tail))
                       for tail in partitions[mask ^ first]}
            sub = mask
            while sub:
                if sub & first and sub in subsets:
                    t = cargo_ids[subsets[sub]]
                    choices.update(tuple(sorted((t,) + tail))
                                   for tail in partitions[mask ^ sub])
                sub = (sub - 1) & mask
            partitions[mask] = choices
        layouts = []
        total = 0
        for partition in sorted(partitions[full]):
            groups = [(t, partition.count(t)) for t in sorted(set(partition))]
            total += math.prod(math.comb(places + multiplicity - 1, multiplicity)
                               for t, multiplicity in groups)
            if total > 180000:
                return None
            layouts.append(groups)
        states = []
        for groups in layouts:
            options = [tuple(itertools.combinations_with_replacement(range(places), m))
                       for t, m in groups]
            for assignment in itertools.product(*options):
                states.append(tuple(t * places + loc
                                    for (t, m), locations in zip(groups, assignment)
                                    for loc in locations))
            if time.perf_counter() >= deadline:
                return None
        index = {key: i for i, key in enumerate(states)}
        goal_key = tuple(sorted(labels[p] * places + goals[p] for p in range(len(weights))))
        goal_index = index[goal_key]
        content = [None] * count + cargo_types
        additions = {}
        for t, cargo in enumerate(content):
            if cargo is not None:
                for c in range(count):
                    merged = tuple(sorted(cargo + (c,)))
                    if merged in cargo_ids:
                        additions[t, c] = cargo_ids[merged]
        targets = []; costs = []; offsets = []
        for i, key in enumerate(states):
            offsets.append(len(targets))
            unique = tuple(dict.fromkeys(key))
            for code in unique:
                t, loc = divmod(code, places)
                if t < count:
                    other = list(key); other.remove(code)
                    child = tuple(sorted(other + [cargo_ids[(t,)] * places + loc]))
                    targets.append(index[child]); costs.append(1)
                    for loaded in unique:
                        lt, ll = divmod(loaded, places)
                        merged = additions.get((lt, t))
                        if ll != loc or merged is None:
                            continue
                        other = list(key); other.remove(code); other.remove(loaded)
                        child = tuple(sorted(other + [merged * places + loc]))
                        targets.append(index[child]); costs.append(1)
                else:
                    cargo = content[t]
                    for c in set(cargo):
                        rest = list(cargo); rest.remove(c)
                        other = list(key); other.remove(code); other.append(c * places + loc)
                        if rest:
                            other.append(cargo_ids[tuple(rest)] * places + loc)
                        targets.append(index[tuple(sorted(other))]); costs.append(1)
                    for dest, cost in enumerate(distances[loc]):
                        if dest == loc or cost == float('inf'):
                            continue
                        other = list(key); other.remove(code); other.append(t * places + dest)
                        targets.append(index[tuple(sorted(other))]); costs.append(int(cost))
            if i % 512 == 0 and time.perf_counter() >= deadline:
                return None
        target = np.asarray(targets, dtype=np.int32)
        edge_cost = np.asarray(costs, dtype=np.int32)
        starts = np.asarray(offsets, dtype=np.int64)
        values = np.zeros(len(states), dtype=np.int32)
        converged = False
        # Starting at zero is important: every finite number of backups remains
        # a LOWER bound. A time-limited iteration from infinity would not be safe.
        for iteration in range(64):
            if time.perf_counter() >= deadline:
                break
            updated = np.minimum.reduceat(edge_cost + values[target], starts)
            updated[goal_index] = 0
            if np.array_equal(updated, values):
                converged = True
                break
            values = updated
        table = cls()
        table.index = index
        table.values = values
        table.labels = labels
        table.cargo_ids = cargo_ids
        table.places = places
        table.converged = converged
        return table

    def lookup(self, positions, trucks, docks, mapping):
        """Project our signed position encoding; ferry membership stays exact."""
        key = []; groups = {}
        for label, pos in zip(self.labels, positions):
            if pos >= 0:
                key.append(label * self.places + mapping[pos])
            else:
                groups.setdefault(-1-pos, []).append(label)
        for truck, cargo in groups.items():
            kind = self.cargo_ids.get(tuple(sorted(cargo)))
            if kind is None:
                return 0
            pos = trucks[truck]
            if pos < 0:
                pos = docks[-1-pos]
            key.append(kind * self.places + mapping[pos])
        index = self.index.get(tuple(sorted(key)))
        return int(self.values[index]) if index is not None else 0


class State:
    """Keep real IDs in positions, but compare interchangeable object shapes.

    No IDs are renamed: each node still generates actions for its own actual
    objects. Equal keys describe a cost-preserving permutation of objects with
    identical capabilities/goals, including the complete containment relation.
    """
    __slots__ = ("positions", "key", "_hash")

    def __init__(self, positions, key):
        object.__setattr__(self, "positions", positions)
        object.__setattr__(self, "key", key)
        object.__setattr__(self, "_hash", hash(key))

    def __setattr__(self, name, value):
        raise AttributeError("immutable state")

    def __getitem__(self, index):
        return self.positions[index]

    def __iter__(self):
        return iter(self.positions)

    def __hash__(self):
        return self._hash

    def __eq__(self, other):
        return isinstance(other, State) and self.key == other.key


class LogisticsProblem(search.Problem):
    """Fixed-order immutable positions; all map data stays outside the state.

    state = (package_positions, truck_positions, ferry_ports).
    Nonnegative positions are locations. For packages, -1-t means inside
    truck index t; for trucks, -1-f means aboard ferry index f. Indices are
    internal only: every action uses the original input IDs.
    """

    def __init__(self, initial):
        self.n = initial["Locations"]
        self.package_ids = tuple(sorted(initial["Packages"]))
        self.truck_ids = tuple(sorted(initial["Trucks"]))
        self.ferry_ids = tuple(sorted(initial["Ferries"]))
        self.weights = tuple(initial["Packages"][p][1] for p in self.package_ids)
        self.goals = tuple(initial["Packages"][p][2] for p in self.package_ids)
        self.capacities = tuple(initial["Trucks"][t][1] for t in self.truck_ids)
        self.ferry_capacities = tuple(initial["Ferries"][f][2] for f in self.ferry_ids)
        self.served = tuple(tuple(sorted(set(initial["Ferries"][f][1])))
                            for f in self.ferry_ids)
        neighbors = [set() for _ in range(self.n)]
        for a, b in initial["Roads"]:
            if a != b:
                neighbors[a].add(b)
                neighbors[b].add(a)
        self.neighbors = tuple(tuple(sorted(ns)) for ns in neighbors)
        self.capable = tuple(tuple(t for t, c in enumerate(self.capacities) if c >= w)
                             for w in self.weights)
        self.max_capacity = max(self.capacities, default=0)
        self.dual_groups = []
        seen_groups = set()
        for k in range(1, len(self.weights) + 1):
            group = tuple(p for p, weight in enumerate(self.weights)
                          if weight * (k+1) > self.max_capacity)
            if group and group not in seen_groups:
                # If every item contributes >= 1/k under weight/C, the
                # weighted workload already dominates this cardinality bound.
                if any(self.weights[p] * k < self.max_capacity for p in group):
                    self.dual_groups.append((group, k))
                seen_groups.add(group)
        self.max_trucks_per_sail = max((min(k, len(self.truck_ids)) for k in self.ferry_capacities), default=0)
        ordered = sorted(self.capacities, reverse=True)
        self.sail_capacity = max((sum(ordered[:k]) for k in self.ferry_capacities
                                  if k > 0), default=0)
        self._build_relaxation()
        classes = sorted(set(zip(self.weights, self.goals)))
        self.package_classes = tuple(classes.index(x) for x in zip(self.weights, self.goals))
        ferry_types = tuple(zip(self.served, self.ferry_capacities))
        types = sorted(set(ferry_types))
        self.ferry_classes = tuple(types.index(x) for x in ferry_types)
        self._build_packing()
        self._build_dead_ports(initial)
        self._build_cuts(initial)
        self._road_side_bits = tuple(sum(1 << edge for edge, (_, _, side) in enumerate(self.road_cuts)
                                        if loc in side) for loc in range(self.n))
        self._all_road_bits = (1 << len(self.road_cuts)) - 1
        paths = {}
        for goal in set(self.goals):
            paths[goal] = tuple(tuple(2*edge + (0 if loc in side else 1)
                                     for edge, (_, _, side) in enumerate(self.road_cuts)
                                     if (loc in side) != (goal in side))
                                for loc in range(self.n))
        self._package_road_paths = tuple(paths[goal] for goal in self.goals)
        self.ferry_regions = tuple(self._components(banned_ferry=(f,))
                                   for f in range(len(self.ferry_ids)))
        self.event_locations = set(self.goals)
        self.event_locations.update(start for start, _, _ in initial["Packages"].values())
        self.event_locations.update(start for start, _ in initial["Trucks"].values())
        for ports in self.served:
            self.event_locations.update(ports)
        self.event_locations.update(v for v, ns in enumerate(self.neighbors) if len(ns) != 2)
        self._build_operators()
        self._build_tree_symmetry()
        self._build_complementary_drives()
        # Bounded memoization only; the supplied search owns its priority queue.
        self._h_cache = OrderedDict()
        self._forest_cache = OrderedDict()
        self._road_cut_cache = OrderedDict()
        self._ferry_cut_cache = OrderedDict()
        self._vehicle_cache = OrderedDict()
        self._state_cargo_cache = OrderedDict()
        self._cargo_drive_cache = OrderedDict()
        self._build_ferry_cache_codes()
        self._build_road_cargo_patterns(initial)
        state = self._state((tuple(initial["Packages"][p][0] for p in self.package_ids),
                 tuple(initial["Trucks"][t][0] for t in self.truck_ids),
                 tuple(initial["Ferries"][f][0] for f in self.ferry_ids)))
        super().__init__(state)

    def _build_complementary_drives(self):
        """Charge only roads outside bridge cuts; ferry links stay free."""
        bridges = {frozenset((a, b)) for a, b, _ in self.road_cuts}
        if all(frozenset((a, b)) in bridges for a, ns in enumerate(self.neighbors) for b in ns):
            self.noncut_distance = None
            return
        distance = [[float('inf')] * self.n for _ in range(self.n)]
        for a, ns in enumerate(self.neighbors):
            distance[a][a] = 0
            for b in ns:
                distance[a][b] = 0 if frozenset((a, b)) in bridges else 1
        for f, ports in enumerate(self.served):
            if self.ferry_capacities[f] > 0:
                for a in ports:
                    for b in ports:
                        distance[a][b] = 0
        for k in range(self.n):
            for row in distance:
                prefix = row[k]
                for b, suffix in enumerate(distance[k]):
                    row[b] = min(row[b], prefix + suffix)
        self.noncut_distance = tuple(tuple(row) for row in distance)

    def _build_road_cargo_patterns(self, initial):
        """Bound consolidation on disjoint parcels and disjoint road costs.

        Adapted from the other branch, with a two-second construction budget.
        Tables remove all empty-vehicle positioning and ferry constraints.
        Selected bridges cost one; every other road and all ferry ports are
        contracted. Real actions project at no greater allocated cost.
        """
        self.road_cargo_patterns = []
        self.pattern_edges = frozenset()
        self.pattern_other_packages = tuple(range(len(self.package_ids)))
        # A table chosen from initial road branches need not be invariant under
        # the separate tree automorphisms used in State equality. Avoid mixing
        # these reductions; existing bounds already handle those symmetric maps.
        if self.tree is not None:
            return
        deadline = time.perf_counter() + 2.0
        used = set(); covered = set()
        classes = {}
        for p, kind in enumerate(self.package_classes):
            classes.setdefault(kind, []).append(p)
        for group in sorted(classes.values(), key=lambda group: -len(group)):
            if len(self.road_cargo_patterns) >= 2 or time.perf_counter() >= deadline:
                break
            if not 2 <= len(group) <= 6 or self.weights[group[0]] <= 0:
                continue
            if 2*self.weights[group[0]] > self.max_capacity:
                continue
            goal = self.goals[group[0]]
            origins = [initial['Packages'][self.package_ids[p]][0] for p in group]
            candidates = {}
            for edge, (_, _, side) in enumerate(self.road_cuts):
                if edge in used:
                    continue
                mask = sum(1 << i for i, loc in enumerate(origins)
                           if (loc in side) != (goal in side))
                if mask:
                    candidates.setdefault(mask, []).append(edge)
            selected = set()
            for i in range(len(group)):
                selected.update(candidates.get(1 << i, ())[:1])
            selected.update(candidates.get((1 << len(group))-1, ())[:2])
            if not selected:
                continue
            parent = list(range(self.n))

            def find(v):
                while parent[v] != v:
                    parent[v] = parent[parent[v]]
                    v = parent[v]
                return v

            def join(a, b):
                parent[find(a)] = find(b)

            blocked = {frozenset(self.road_cuts[e][:2]) for e in selected}
            for a, ns in enumerate(self.neighbors):
                for b in ns:
                    if frozenset((a, b)) not in blocked:
                        join(a, b)
            for f, ports in enumerate(self.served):
                if self.ferry_capacities[f] > 0:
                    for port in ports[1:]:
                        join(ports[0], port)
            roots = {find(goal): 0}
            for e in sorted(selected):
                for v in self.road_cuts[e][:2]:
                    roots.setdefault(find(v), len(roots))
            mapping = tuple(roots.get(find(v), 0) for v in range(self.n))
            size = len(roots)
            if size > 7:
                continue
            distance = [[float('inf')] * size for _ in range(size)]
            for v in range(size):
                distance[v][v] = 0
            for e in selected:
                a, b = (mapping[v] for v in self.road_cuts[e][:2])
                distance[a][b] = distance[b][a] = 1
            for k in range(size):
                for a in range(size):
                    for b in range(size):
                        distance[a][b] = min(distance[a][b], distance[a][k] + distance[k][b])
            table = _CargoPattern.build(tuple(self.weights[p] for p in group),
                                        tuple(mapping[self.goals[p]] for p in group),
                                        size, distance, self.max_capacity, deadline)
            if table is not None:
                used.update(selected); covered.update(group)
                self.road_cargo_patterns.append((table, mapping, tuple(group), frozenset(selected)))
        self.pattern_edges = frozenset(used)
        self.pattern_other_packages = tuple(p for p in range(len(self.package_ids)) if p not in covered)

    def _noncut_cargo_bound(self, cargo):
        """DRIVEs carrying cargo on non-bridge roads, never empty approaches."""
        if self.noncut_distance is None or not cargo:
            return 0
        solo = shared = 0
        for p, loc in cargo:
            distance = self.noncut_distance[loc][self.goals[p]]
            if all(self.weights[p] + self.weights[q] > self.max_capacity
                   for q, _ in cargo if q != p):
                solo += distance
            else:
                shared = max(shared, distance)
        return solo + shared

    def _cargo_drive_bound(self, cargo):
        """Only loaded DRIVEs: safe to add to an exclusive parcel's trip."""
        key = tuple(cargo)
        cached = self._cargo_drive_cache.get(key)
        if cached is not None:
            self._cargo_drive_cache.move_to_end(key)
            return cached
        value = self._noncut_cargo_bound(cargo)
        payload = [0] * (2*len(self.road_cuts))
        for p, loc in cargo:
            for direction in self._package_road_paths[p][loc]:
                payload[direction] |= 1 << p
        value += sum(self._loads(mask) for mask in payload)
        if len(self._cargo_drive_cache) >= 8192:
            self._cargo_drive_cache.popitem(last=False)
        self._cargo_drive_cache[key] = value
        return value

    def _state(self, positions):
        packages, trucks, ferries = positions
        cached = self._state_cargo_cache.get(packages)
        if cached is None:
            cargo = [[] for _ in trucks]
            ground = []
            for p, pos in enumerate(packages):
                if pos < 0:
                    cargo[-1-pos].append(self.package_classes[p])
                else:
                    ground.append((self.package_classes[p], pos))
            cargo = tuple(tuple(sorted(contents)) for contents in cargo)
            ground = tuple(sorted(ground))
            cached = (cargo, ground)
            if len(self._state_cargo_cache) >= 16384:
                self._state_cargo_cache.popitem(last=False)
            self._state_cargo_cache[packages] = cached
        else:
            self._state_cargo_cache.move_to_end(packages)
            cargo, ground = cached
        aboard = [[] for _ in ferries]
        standing = []
        for t, pos in enumerate(trucks):
            contents = cargo[t]
            if pos < 0:
                aboard[-1-pos].append((self.capacities[t], contents))
            else:
                standing.append((self.capacities[t], pos, contents))
        if self.tree is not None:
            locations = self._tree_mapping(ground, standing)
            ground = [(kind, locations[pos]) for kind, pos in ground]
            standing = [(capacity, locations[pos], cargo) for capacity, pos, cargo in standing]
        ships = tuple(sorted((self.ferry_classes[f], port, tuple(sorted(aboard[f])))
                             for f, port in enumerate(ferries)))
        return State(positions, (tuple(sorted(ground)), tuple(sorted(standing)), ships))

    def _build_tree_symmetry(self):
        """Road-forest automorphisms fixing every goal and every served port.

        Rooting each component selects a subgroup of its automorphisms; no
        symmetry is guessed on cyclic maps. Only equal-shaped child branches
        are permuted. Fixed ports also fix all ferry relations and positions.
        """
        self.tree = None
        if self.n > 32:
            return
        fixed = set(self.goals)
        for ports in self.served:
            fixed.update(ports)
        seen = set(); roots = []; order = []; children = [[] for _ in range(self.n)]
        for start in range(self.n):
            if start in seen:
                continue
            component = []; stack = [(start, -1)]; seen.add(start)
            while stack:
                v, parent = stack.pop(); component.append(v)
                for w in self.neighbors[v]:
                    if w == parent:
                        continue
                    if w in seen:
                        return
                    seen.add(w); stack.append((w, v))
            # A fixed vertex is an invariant root. Otherwise a tree center
            # provides a static root; fixing it is still a valid subgroup.
            choices = sorted(fixed.intersection(component))
            if choices:
                root = choices[0]
            else:
                remaining = set(component)
                degree = {v: len(self.neighbors[v]) for v in component}
                leaves = [v for v in component if degree[v] <= 1]
                while len(remaining) > 2:
                    remaining.difference_update(leaves)
                    next_leaves = []
                    for v in leaves:
                        for w in self.neighbors[v]:
                            if w in remaining:
                                degree[w] -= 1
                                if degree[w] == 1:
                                    next_leaves.append(w)
                    leaves = next_leaves
                root = min(remaining)
            roots.append(root)
            stack = [(root, -1)]
            while stack:
                v, parent = stack.pop(); order.append(v)
                children[v] = [w for w in self.neighbors[v] if w != parent]
                stack.extend((w, v) for w in children[v])
        shapes = [None] * self.n
        ambiguous = False
        for v in reversed(order):
            kinds = tuple(sorted(shapes[w] for w in children[v]))
            ambiguous |= len(set(kinds)) < len(kinds)
            shapes[v] = (v if v in fixed else (-2 if v in self.event_locations else -1), kinds)
        if ambiguous:
            targets = [tuple(sorted(cs, key=lambda w: (shapes[w], w))) for cs in children]
            self.tree = (roots, order, children, shapes, targets)

    def _tree_mapping(self, ground, standing):
        roots, order, children, shapes, targets = self.tree
        local_packages = [[] for _ in range(self.n)]
        local_trucks = [[] for _ in range(self.n)]
        for kind, pos in ground:
            local_packages[pos].append(kind)
        for capacity, pos, cargo in standing:
            local_trucks[pos].append((capacity, cargo))
        dynamic = [None] * self.n
        for v in reversed(order):
            dynamic[v] = (tuple(sorted(local_packages[v])), tuple(sorted(local_trucks[v])),
                          tuple(sorted((shapes[w], dynamic[w]) for w in children[v])))
        mapping = list(range(self.n))
        stack = [(v, v) for v in roots]
        while stack:
            source, target = stack.pop()
            mapping[source] = target
            ordered = sorted(children[source], key=lambda w: (shapes[w], dynamic[w]))
            stack.extend(zip(ordered, targets[target]))
        return mapping

    def path_cost(self, c, state1, action, state2):
        """Do not change this: SAIL costs 3, every other action costs 1."""
        return c + (3 if action.startswith("SAIL") else 1)

    def _build_relaxation(self):
        """Floyd-Warshall on a simplified map, never the real state space.

        Vertices are standing locations and (ferry, port) aboard positions.
        Ferry availability, truck occupancy, and package transfers are relaxed.
        Five metrics count all movement, drives, boarding/disembarking, sails,
        and non-sailing movement. Each ignores constraints, never adds them.
        """
        self.aboard_vertex = []
        size = self.n
        for ports in self.served:
            vertices = {}
            for port in ports:
                vertices[port] = size
                size += 1
            self.aboard_vertex.append(vertices)
        edges = []
        for a, ns in enumerate(self.neighbors):
            for b in ns:
                edges.append((a, b, (1, 1, 0, 0, 1)))
        for f, ports in enumerate(self.served):
            if self.ferry_capacities[f] <= 0:
                continue
            vertices = self.aboard_vertex[f]
            for a in ports:
                edges.append((a, vertices[a], (1, 0, 1, 0, 1)))
                edges.append((vertices[a], a, (1, 0, 1, 0, 1)))
                for b in ports:
                    if a != b:
                        edges.append((vertices[a], vertices[b], (3, 0, 0, 1, 0)))
        tables = []
        inf = float("inf")
        for metric in range(5):
            distance = [[inf] * size for _ in range(size)]
            for a in range(size):
                distance[a][a] = 0
            for a, b, costs in edges:
                distance[a][b] = min(distance[a][b], costs[metric])
            for k in range(size):
                via = distance[k]
                for row in distance:
                    prefix = row[k]
                    if prefix == inf:
                        continue
                    for b, suffix in enumerate(via):
                        if prefix + suffix < row[b]:
                            row[b] = prefix + suffix
            tables.append(tuple(tuple(row) for row in distance))
        (self.movement, self.drive_distance, self.transfer_distance,
         self.sail_distance, self.non_sail_distance) = tables

    def _build_packing(self):
        """Static bin-packing relaxation, not logistics-state exploration.

        At most 16 packages: subset DP keeps (fewest bins, smallest final load).
        Any crossing carrying a subset needs at least this many max-size trucks.
        Larger inputs use cheaper weight/count lower bounds without a DP table.
        """
        self.packing = None
        self.mask_weights = None
        if len(self.weights) > 16:
            return
        size = 1 << len(self.weights)
        best = [(0, 0)] * size
        weights = [0] * size
        for mask in range(1, size):
            low = mask & -mask
            weights[mask] = weights[mask ^ low] + self.weights[low.bit_length()-1]
            candidate = (len(self.weights) + 1, 0)
            bits = mask
            while bits:
                bit = bits & -bits
                bits -= bit
                p = bit.bit_length() - 1
                previous = mask ^ bit
                bins, load = best[previous]
                weight = self.weights[p]
                option = ((bins, load + weight) if previous and load + weight <= self.max_capacity
                          else (bins + 1, weight))
                candidate = min(candidate, option)
            best[mask] = candidate
        self.packing = tuple(bins for bins, _ in best)
        self.mask_weights = tuple(weights)

    def _loads(self, mask):
        if not mask:
            return 0
        if self.packing is not None:
            return self.packing[mask]
        weights = [w for p, w in enumerate(self.weights) if mask & (1 << p)]
        capacity = self.max_capacity
        if capacity <= 0:
            return 1
        return max(1, (sum(weights) + capacity - 1) // capacity,
                   sum(2 * w > capacity for w in weights))

    def _mask_weight(self, mask):
        if self.mask_weights is not None:
            return self.mask_weights[mask]
        return sum(weight for p, weight in enumerate(self.weights) if mask & (1 << p))

    def _components(self, banned_road=None, banned_ferry=None):
        neighbors = [set() for _ in range(self.n)]
        for a, ns in enumerate(self.neighbors):
            for b in ns:
                if (min(a, b), max(a, b)) != banned_road:
                    neighbors[a].add(b)
        banned = set(banned_ferry or ())
        for f, ports in enumerate(self.served):
            if f not in banned and self.ferry_capacities[f] > 0:
                for a in ports:
                    neighbors[a].update(ports)
        labels = [-1] * self.n
        count = 0
        for start in range(self.n):
            if labels[start] != -1:
                continue
            labels[start] = count
            stack = [start]
            while stack:
                for b in neighbors[stack.pop()]:
                    if labels[b] == -1:
                        labels[b] = count
                        stack.append(b)
            count += 1
        return tuple(labels)

    def _build_dead_ports(self, initial):
        """An empty island served by one service group is no interchange.

        Visiting it cannot deliver/pick up anything or change ferry services.
        Since each ferry sails directly between all its ports, such a detour
        can be deleted. Initially parked empty ferries need no final position.
        """
        important = set(self.goals)
        important.update(start for start, _, _ in initial["Packages"].values())
        important.update(start for start, _ in initial["Trucks"].values())
        seen = set(); dead = set()
        for start in range(self.n):
            if start in seen:
                continue
            island = {start}; stack = [start]; seen.add(start)
            while stack:
                for v in self.neighbors[stack.pop()]:
                    if v not in seen:
                        island.add(v); seen.add(v); stack.append(v)
            services = {ports for f, ports in enumerate(self.served)
                        if self.ferry_capacities[f] > 0 and not island.isdisjoint(ports)}
            if island.isdisjoint(important) and len(services) <= 1:
                dead.update(island)
        self.dead_ports = frozenset(dead)
        self.useful_ferries = frozenset(f for f, ports in enumerate(self.served)
                                       if self.ferry_capacities[f] > 0
                                       and sum(port not in dead for port in ports) >= 2)

    def _build_cuts(self, initial):
        """Find roads/ferries whose removal disconnects the relaxed map."""
        self.components = self._components()
        self.component_count = len(set(self.components))
        self.road_cuts = []
        self.unneeded_drives = {}
        useful = set(self.goals)
        useful.update(start for start, _, _ in initial["Packages"].values())
        for f, ports in enumerate(self.served):
            if self.ferry_capacities[f] > 0:
                useful.update(ports)
        # A truck may start inside a useless road branch and must be allowed to
        # leave it. Only entering a branch with no package/goal/port is removed.
        for a, ns in enumerate(self.neighbors):
            for b in ns:
                if a < b:
                    labels = self._components(banned_road=(a, b))
                    if labels[a] != labels[b]:
                        side = frozenset(v for v in range(self.n) if labels[v] == labels[a])
                        self.road_cuts.append((a, b, side))
                        if side.isdisjoint(useful):
                            self.unneeded_drives[(b, a)] = side
                        other_side = frozenset(v for v in range(self.n) if labels[v] == labels[b])
                        if other_side.isdisjoint(useful):
                            self.unneeded_drives[(a, b)] = other_side
        self.ferry_cuts = []
        groups = {}
        for f, ports in enumerate(self.served):
            if self.ferry_capacities[f] <= 0:
                continue
            groups.setdefault(ports, []).append(f)
        for ports, members in groups.items():
            labels = self._components(banned_ferry=members)
            regions = tuple(sorted({labels[p] for p in ports}))
            if len(regions) > 1:
                self.ferry_cuts.append((frozenset(members), labels, regions))

    def _build_ferry_cache_codes(self):
        """Injective multiset digits for the ferry-cut cache projections.

        Every digit has room for ALL packages/trucks/ferries, so addition can
        never carry into another category. This encodes the previous sorted
        tuples exactly, including empty trucks' aboard/standing status.
        """
        self._ferry_cache_codes = {}
        bits = max(len(self.package_ids), len(self.truck_ids), len(self.ferry_ids), 1).bit_length()
        capacities = {c: i for i, c in enumerate(sorted(set(self.capacities)))}
        for members, regions, _ in self.ferry_cuts:
            size = max(regions) + 1
            signatures = sorted(set((w, regions[g]) for w, g in zip(self.weights, self.goals)))
            classes = {signature: i for i, signature in enumerate(signatures)}
            cargo = tuple(tuple((1 << ((classes[w, regions[g]]*size*2 + regions[loc]*2)*bits),
                                 1 << ((classes[w, regions[g]]*size*2 + regions[loc]*2+1)*bits))
                                for loc in range(self.n))
                          for w, g in zip(self.weights, self.goals))
            trucks = tuple(tuple((1 << ((capacities[c]*size*2 + regions[loc]*2)*bits),
                                  1 << ((capacities[c]*size*2 + regions[loc]*2+1)*bits))
                                 for loc in range(self.n)) for c in self.capacities)
            ships = tuple(1 << (regions[loc]*bits) for loc in range(self.n))
            self._ferry_cache_codes[members] = cargo, trucks, ships

    def _cut_bounds(self, state, required):
        """Separate unavoidable DRIVE, BOARD/DISEMBARK and SAIL counts.

        Each road cut belongs to a different drive edge. Each ferry cut counts
        only actions of that ferry. Thus counts within each family may be added;
        they replace (rather than add to) other bounds on the same action types.
        """
        packages, trucks, ferries = state
        truck_locations = tuple(pos if pos >= 0 else ferries[-1-pos] for pos in trucks)
        remaining = []
        for p, pos in enumerate(packages):
            if pos == self.goals[p]:
                continue
            loc = pos if pos >= 0 else truck_locations[-1-pos]
            aboard = trucks[-1-pos] if pos < 0 else 0
            remaining.append((1 << p, loc, self.goals[p], -1-aboard if aboard < 0 else None))
        grounded = tuple(sorted((self.weights[p], pos) for p, pos in enumerate(packages)
                                if pos >= 0 and pos != self.goals[p]))
        # Road equations use effective locations, capable-truck capacities,
        # ground pickups and required visits, but never ferry containment.
        # Keep this cache separate from ferry equations, where empty aboard
        # trucks MUST remain distinguishable from standing trucks.
        road_key = (tuple((bit, loc) for bit, loc, _, _ in remaining),
                    tuple(sorted(zip(self.capacities, truck_locations))),
                    frozenset(required), grounded)
        road_result = self._road_cut_cache.get(road_key)
        if road_result is None:
            drives = 0
            edge_drives = []
            balance = [0] * self.n
            for loc in truck_locations:
                balance[loc] += 1
            payload = [0] * (2*len(self.road_cuts))
            for bit, loc, _, _ in remaining:
                for direction in self._package_road_paths[bit.bit_length()-1][loc]:
                    payload[direction] |= bit
            required_left = required_right = 0
            for loc in required:
                sides = self._road_side_bits[loc]
                required_left |= sides
                required_right |= self._all_road_bits ^ sides
            # A nearby undersized truck cannot collect a parcel. Every capable
            # truck being outside a pickup's side forces a reverse crossing even
            # when some smaller truck is already there. This counts the same road
            # action as other cut requirements, so combine with max, never addition.
            eligible_sides = {}
            forced_out = forced_in = 0
            for p, pos in enumerate(packages):
                if pos < 0 or pos == self.goals[p]:
                    continue
                weight = self.weights[p]
                sides = eligible_sides.get(weight)
                if sides is None:
                    left = right = 0
                    for t in self.capable[p]:
                        bits = self._road_side_bits[truck_locations[t]]
                        left |= bits
                        right |= self._all_road_bits ^ bits
                    sides = (left, right)
                    eligible_sides[weight] = sides
                pickup = self._road_side_bits[pos]
                forced_in |= pickup & ~sides[0]
                forced_out |= (self._all_road_bits ^ pickup) & ~sides[1]
            for edge, (a, b, side) in enumerate(self.road_cuts):
                outbound, inbound = payload[2*edge], payload[2*edge+1]
                out = self._loads(outbound)
                into = self._loads(inbound)
                stock = sum(loc in side for loc in truck_locations)
                other_stock = len(trucks) - stock
                if (not stock and required_left & (1 << edge)) or forced_in & (1 << edge):
                    into = max(into, 1)
                if (not other_stock and required_right & (1 << edge)) or forced_out & (1 << edge):
                    out = max(out, 1)
                # Truck conservation: initial stock + arrivals >= departures.
                into = max(into, out - stock)
                out = max(out, into - other_stock)
                drives += out + into
                edge_drives.append(out + into)
                balance[a] += into - out
                balance[b] += out - into
            # The compulsory road flows can demand more trucks at a junction than
            # arrive there. Remaining actions must transport trucks from surplus
            # vertices to deficit vertices. Letting every deficit reuse the nearest
            # surplus ignores supply limits, so this is a cheap extra lower bound.
            # Ferry moves are free in this metric and cannot be counted twice.
            supply = [v for v, amount in enumerate(balance) if amount > 0]
            if supply:
                for v, amount in enumerate(balance):
                    if amount < 0:
                        drives += -amount * min(self.drive_distance[s][v] for s in supply)
            road_result = (drives, tuple(edge_drives))
            if len(self._road_cut_cache) >= 32768:
                self._road_cut_cache.popitem(last=False)
            self._road_cut_cache[road_key] = road_result
        else:
            self._road_cut_cache.move_to_end(road_key)
        drives, edge_drives = road_result
        transfers = sails = 0
        for members, regions, served_regions in self.ferry_cuts:
            # Only regions matter to this group's ferry conservation equations.
            # Keep cargo weights/current regions/goal regions and whether cargo
            # or even an EMPTY truck is already aboard a member ferry. Exact
            # road positions and membership in other ferry groups are relaxed.
            cargo_codes, truck_codes, ship_codes = self._ferry_cache_codes[members]
            cargo_code = sum(cargo_codes[bit.bit_length()-1][loc][aboard in members]
                             for bit, loc, _, aboard in remaining)
            truck_code = sum(truck_codes[t][loc][trucks[t] < 0 and -1-trucks[t] in members]
                             for t, loc in enumerate(truck_locations))
            ferry_key = (members, cargo_code, truck_code,
                         sum(ship_codes[ferries[f]] for f in members))
            cached_ferry = self._ferry_cut_cache.get(ferry_key)
            if cached_ferry is not None:
                self._ferry_cut_cache.move_to_end(ferry_key)
                transfers += cached_ferry[0]
                sails += cached_ferry[1]
                continue
            previous_transfers = transfers
            boards = {}; disembarks = {}; departures = {}; arrivals = {}
            for bit, loc, goal, aboard in remaining:
                source, target = regions[loc], regions[goal]
                if source != target:
                    departures[source] = departures.get(source, 0) | bit
                    arrivals[target] = arrivals.get(target, 0) | bit
                    if aboard not in members:
                        boards[source] = boards.get(source, 0) | bit
                if source != target or aboard in members:
                    disembarks[target] = disembarks.get(target, 0) | bit
            board_counts = {r: self._loads(mask) for r, mask in boards.items()}
            disembark_counts = {r: self._loads(mask) for r, mask in disembarks.items()}
            available = {r: 0 for r in served_regions}
            available_capacity = {r: 0 for r in served_regions}
            already_aboard = 0
            for t, pos in enumerate(trucks):
                if pos < 0 and -1-pos in members:
                    already_aboard += 1
                else:
                    r = regions[truck_locations[t]]
                    if r in available:
                        available[r] += 1
                        available_capacity[r] += self.capacities[t]
            # A pickup region cannot send more trucks than it initially has
            # plus arrivals. This can force empty truck returns even when no
            # package has that region as its destination.
            completed = 0
            for r in served_regions:
                capacity_shortage = self._mask_weight(boards.get(r, 0)) - available_capacity[r]
                arrivals_for_weight = ((max(0, capacity_shortage) + self.max_capacity - 1)
                                       // self.max_capacity) if self.max_capacity > 0 else 0
                completed += max(disembark_counts.get(r, 0),
                                 board_counts.get(r, 0) - available[r], arrivals_for_weight)
            new_boards = sum(board_counts.values())
            completed = max(completed, new_boards)
            new_boards = max(new_boards, completed - already_aboard)
            transfers += new_boards + completed
            capacity = max(self.ferry_capacities[f] for f in members)
            out = {r: (self._loads(mask) + capacity - 1) // capacity
                   for r, mask in departures.items()}
            into = {r: (self._loads(mask) + capacity - 1) // capacity
                    for r, mask in arrivals.items()}
            needed = max(sum(out.values()), sum(into.values()))
            # Every newly required ride sails at least once in an optimal
            # plan; BOARD/DISEMBARK without sailing is a removable identity.
            needed = max(needed, (new_boards + capacity - 1) // capacity)
            stock = {r: sum(regions[ferries[f]] == r for f in members) for r in served_regions}
            needed = max(needed, sum(max(into.get(r, 0), out.get(r, 0) - stock[r])
                                     for r in served_regions))
            if len(served_regions) == 2:
                a, b = served_regions
                forward, backward = out.get(a, 0), out.get(b, 0)
                # Conservation includes empty returns, even with several ships.
                backward = max(backward, forward - stock[a])
                forward = max(forward, backward - stock[b])
                needed = max(needed, forward + backward)
            sails += needed
            if len(self._ferry_cut_cache) >= 16384:
                self._ferry_cut_cache.popitem(last=False)
            self._ferry_cut_cache[ferry_key] = (transfers - previous_transfers, needed)
        result = (drives, transfers, sails, tuple(edge_drives))
        return result

    def _build_operators(self):
        """Ground action preconditions/effects for strong stubborn sets.

        Every action writes one position variable. Capacity guards additionally
        couple LOAD/UNLOAD on a truck and BOARD/DISEMBARK on a ferry. These
        resource conflicts are included even when the position effects differ.
        This is an action-dependency table, not a replacement search algorithm.
        """
        P, T = len(self.package_ids), len(self.truck_ids)
        operators = []
        self.achievers = {}
        self.leavers = {}
        self.readers = {}
        self.writers = {}
        self.resources = {}
        self.goal_operators = [[] for _ in self.package_ids]
        self.unload_operators = [[] for _ in self.truck_ids]
        self.disembark_operators = [[] for _ in self.ferry_ids]

        def add(action, pre, effect, resource=None, guard=None, package=None):
            index = len(operators)
            operators.append((action, tuple(pre), effect, resource, guard, package))
            self.achievers.setdefault(effect, []).append(index)
            self.writers.setdefault(effect[0], set()).add(index)
            for var, _ in pre:
                self.readers.setdefault(var, set()).add(index)
            for var, value in pre:
                if var == effect[0] and value != effect[1]:
                    self.leavers.setdefault((var, value), []).append(index)
            if resource is not None:
                self.resources.setdefault(resource, set()).add(index)
            return index

        for p, pid in enumerate(self.package_ids):
            for t in self.capable[p]:
                tid = self.truck_ids[t]
                for loc in sorted(self.event_locations):
                    add(f"LOAD{{{pid},{tid}}}", ((p, loc), (P+t, loc)),
                        (p, -1-t), ("truck", t), ("load", p, t), p)
                    op = add(f"UNLOAD{{{pid},{tid}}}", ((p, -1-t), (P+t, loc)),
                             (p, loc), ("truck", t), None, p)
                    self.unload_operators[t].append(op)
                    if loc == self.goals[p]:
                        self.goal_operators[p].append(op)
        for t, tid in enumerate(self.truck_ids):
            for a, ns in enumerate(self.neighbors):
                for b in ns:
                    if (a, b) not in self.unneeded_drives:
                        add(f"DRIVE{{{tid},{a},{b}}}", ((P+t, a),), (P+t, b))
            for f, fid in enumerate(self.ferry_ids):
                if f not in self.useful_ferries:
                    continue
                for port in self.served[f]:
                    if port in self.dead_ports:
                        continue
                    add(f"BOARD{{{tid},{fid}}}", ((P+t, port), (P+T+f, port)),
                        (P+t, -1-f), ("ferry", f), ("board", t, f))
                    op = add(f"DISEMBARK{{{tid},{fid}}}", ((P+t, -1-f), (P+T+f, port)),
                             (P+t, port), ("ferry", f))
                    self.disembark_operators[f].append(op)
        for f, fid in enumerate(self.ferry_ids):
            if f not in self.useful_ferries:
                continue
            for a in self.served[f]:
                for b in self.served[f]:
                    if a != b and b not in self.dead_ports:
                        add(f"SAIL{{{fid},{a},{b}}}", ((P+T+f, a),), (P+T+f, b))
        self.operators = operators
        self._interference = {}

    def _interfering(self, index):
        cached = self._interference.get(index)
        if cached is not None:
            return cached
        _, pre, effect, resource, guard, _ = self.operators[index]
        candidates = set(self.writers.get(effect[0], ()))
        candidates.update(self.readers.get(effect[0], ()))
        for var, _ in pre:
            candidates.update(self.writers.get(var, ()))
        if resource is not None:
            candidates.update(self.resources[resource])
        facts = dict(pre)
        result = []
        for other in candidates:
            if other == index:
                continue
            _, other_pre, other_effect, other_resource, other_guard, _ = self.operators[other]
            # Operators with contradictory position preconditions cannot be
            # jointly applicable and therefore cannot interfere in any state.
            if any(var in facts and facts[var] != value for var, value in other_pre):
                continue
            conflict = (effect[0] == other_effect[0] and effect[1] != other_effect[1])
            conflict |= any(var == effect[0] and value != effect[1] for var, value in other_pre)
            conflict |= any(var == other_effect[0] and value != other_effect[1] for var, value in pre)
            # Releasing capacity commutes with consuming it when both actions
            # are already applicable. Only two consumers can disable each
            # other: LOAD/LOAD or BOARD/BOARD. Position conflicts above still
            # cover movement and actions on the same package/truck.
            resource_conflict = (resource is not None and resource == other_resource
                                 and guard is not None and other_guard is not None)
            if resource_conflict and guard[0] == "load":
                resource_conflict = (self.weights[guard[1]] > 0 and self.weights[other_guard[1]] > 0
                                     and sum(self.weights) > self.capacities[guard[2]])
            conflict |= resource_conflict
            if conflict:
                result.append(other)
        cached = tuple(result)
        self._interference[index] = cached
        return cached

    def _stubborn_actions(self, state, loads, occupancy, active):
        """Close a goal landmark under necessary enablers and interference.

        An absent position fact needs one of its achievers. A full truck needs
        an unload; a full ferry needs a disembark. Applicable actions require
        all potentially interfering operators. Thus some optimal first action
        survives, without imposing an arbitrary order on dependent deliveries.
        """
        if len(self.operators) > 5000:
            return None
        packages, trucks, ferries = state
        unfinished = [p for p, pos in enumerate(packages) if pos != self.goals[p]
                      and (active is None or self.components[self.goals[p]] == active)]
        if not unfinished:
            return None
        # A single unsatisfied package goal is a disjunctive action landmark.
        p = min(unfinished, key=lambda p: len(self.goal_operators[p]))
        selected = set(self.goal_operators[p])
        pending = list(selected)
        values = packages + trucks + ferries
        allowed = set()
        processed = 0
        while pending:
            processed += 1
            # Large closures prune little and can cost more than expansion.
            # Abandoning reduction restores the complete safe successor set.
            if processed > 96:
                return None
            index = pending.pop()
            action, pre, _, _, guard, package = self.operators[index]
            # Moving a delivered package is unnecessary in every optimal plan.
            if package is not None and packages[package] == self.goals[package]:
                continue
            additions = None
            for var, value in pre:
                if values[var] != value:
                    achieve = self.achievers.get((var, value), ())
                    leave = self.leavers.get((var, values[var]), ())
                    option = achieve if len(achieve) < len(leave) else leave
                    if additions is None or len(option) < len(additions):
                        additions = option
            if additions is not None:
                # Either the target fact must be achieved, or the variable
                # must first leave its current value. Both are necessary sets;
                # the smaller one often avoids reasoning about distant actions.
                pass
            elif guard is not None and guard[0] == "load" and loads[guard[2]] + self.weights[guard[1]] > self.capacities[guard[2]]:
                t = guard[2]
                additions = [i for i in self.unload_operators[t]
                             if packages[self.operators[i][5]] == -1-t
                             and self.weights[self.operators[i][5]] > 0]
            elif guard is not None and guard[0] == "board" and occupancy[guard[2]] >= self.ferry_capacities[guard[2]]:
                f = guard[2]
                additions = [i for i in self.disembark_operators[f]
                             if values[self.operators[i][1][0][0]] == -1-f]
            else:
                allowed.add(action)
                additions = self._interfering(index)
            for other in additions:
                if other not in selected:
                    selected.add(other)
                    pending.append(other)
        # Empty means an impossible goal or an overly restricted envelope.
        # Falling back to all legal actions is conservative in either case.
        return allowed or None

    def goal_test(self, state):
        return state[0] == self.goals

    def successor(self, state):
        """Only exchange cargo at event locations.

        A degree-2 corridor has no initial truck/package, goal, port or junction
        in its interior. Round every exchange coordinate in that corridor to
        an endpoint using a common threshold. Expected total driving distance
        is unchanged (for each segment E|round(x)-round(y)| = |x-y|); therefore
        some threshold does not increase cost. Colocation, cargo order and all
        capacities are preserved. Thus an optimal INITIAL plan exists with no
        interior exchanges. This reduction need not preserve the optimum from
        an arbitrary externally supplied intermediate state.
        """
        packages, trucks, ferries = state
        # A transit truck cannot exchange cargo or board inside a corridor.
        # Its drives commute with every other object's actions. Finish that
        # transit before starting unrelated work; keep both directions so no
        # route choice is imposed. A* still executes/costs each individual road.
        for t, pos in enumerate(trucks):
            if pos >= 0 and pos not in self.event_locations:
                out = {}
                for target in self.neighbors[pos]:
                    changed = trucks[:t] + (target,) + trucks[t+1:]
                    child = self._state((packages, changed, ferries))
                    out[child] = (f"DRIVE{{{self.truck_ids[t]},{pos},{target}}}", child)
                return list(out.values())
        # While a ferry is full, other trucks cannot board it. Sailing and
        # its first disembark commute with all outside actions. Every BOARD in
        # an optimal initial plan eventually has a DISEMBARK: otherwise that
        # BOARD can be deleted (final truck positions do not matter). Thus this
        # also serializes useful empty-truck trips without losing a root plan.
        for f, capacity in enumerate(self.ferry_capacities):
            aboard = [t for t, pos in enumerate(trucks) if pos == -1-f]
            if capacity <= 0 or not aboard:
                continue
            if len(aboard) < capacity:
                # The same first-step reduction is safe if no outside truck
                # can reach this dock without moving this very ferry. After a
                # sail we reconsider, permitting boarding at the arrival port.
                regions = self.ferry_regions[f]
                dock = regions[ferries[f]]
                if any(pos != -1-f and regions[pos if pos >= 0 else ferries[-1-pos]] == dock
                       for pos in trucks):
                    continue
            out = {}
            for t in aboard:
                changed = trucks[:t] + (ferries[f],) + trucks[t+1:]
                child = self._state((packages, changed, ferries))
                out[child] = (f"DISEMBARK{{{self.truck_ids[t]},{self.ferry_ids[f]}}}", child)
            for target in self.served[f]:
                if target != ferries[f] and target not in self.dead_ports:
                    changed = ferries[:f] + (target,) + ferries[f+1:]
                    child = self._state((packages, trucks, changed))
                    out[child] = (f"SAIL{{{self.ferry_ids[f]},{ferries[f]},{target}}}", child)
            return list(out.values())
        active = None
        if self.component_count > 1:
            active = min((self.components[self.goals[p]] for p, pos in enumerate(packages)
                          if pos != self.goals[p]), default=None)
        # Delivery can be moved before any other action: it frees capacity and
        # the package has no further work. Keep one optimal action ordering.
        for p, pos in enumerate(packages):
            if active is not None and self.components[self.goals[p]] != active:
                continue
            if pos < 0 and trucks[-1-pos] == self.goals[p]:
                t = -1-pos
                changed = packages[:p] + (self.goals[p],) + packages[p+1:]
                return [(f"UNLOAD{{{self.package_ids[p]},{self.truck_ids[t]}}}",
                         self._state((changed, trucks, ferries)))]
        loads = [0] * len(trucks)
        occupancy = [0] * len(ferries)
        for p, pos in enumerate(packages):
            if pos < 0:
                loads[-1 - pos] += self.weights[p]
        for pos in trucks:
            if pos < 0:
                occupancy[-1 - pos] += 1
        pickups = {pos for p, pos in enumerate(packages) if pos >= 0 and pos != self.goals[p]}
        allowed = self._stubborn_actions(state, loads, occupancy, active)
        out = {}

        def add(action, positions):
            if allowed is not None and action not in allowed:
                return
            child = self._state(positions)
            if child == state:
                return
            previous = out.get(child)
            if previous is None or (previous[0].startswith("SAIL") and not action.startswith("SAIL")):
                out[child] = (action, child)

        def package_move(p, target, action):
            changed = packages[:p] + (target,) + packages[p + 1:]
            add(action, (changed, trucks, ferries))

        def truck_move(t, target, action):
            changed = trucks[:t] + (target,) + trucks[t + 1:]
            add(action, (packages, changed, ferries))

        for p, pos in enumerate(packages):
            if active is not None and self.components[self.goals[p]] != active:
                continue
            pid = self.package_ids[p]
            if pos >= 0:
                # A delivered package never needs moving again: removing all
                # its later transport only frees capacity and preserves the goal.
                if pos == self.goals[p]:
                    continue
                for t in self.capable[p]:
                    if trucks[t] == pos and loads[t] + self.weights[p] <= self.capacities[t]:
                        package_move(p, -1-t, f"LOAD{{{pid},{self.truck_ids[t]}}}")
            else:
                t = -1-pos
                if trucks[t] in self.event_locations:
                    package_move(p, trucks[t], f"UNLOAD{{{pid},{self.truck_ids[t]}}}")
        for t, pos in enumerate(trucks):
            tid = self.truck_ids[t]
            loc = pos if pos >= 0 else ferries[-1-pos]
            if active is not None and self.components[loc] != active:
                continue
            if pos >= 0:
                for target in self.neighbors[pos]:
                    far = self.unneeded_drives.get((pos, target))
                    if allowed is not None or far is None or not far.isdisjoint(pickups):
                        truck_move(t, target, f"DRIVE{{{tid},{pos},{target}}}")
                for f, port in enumerate(ferries):
                    if f in self.useful_ferries and pos == port and occupancy[f] < self.ferry_capacities[f]:
                        truck_move(t, -1-f, f"BOARD{{{tid},{self.ferry_ids[f]}}}")
            else:
                f = -1-pos
                truck_move(t, ferries[f], f"DISEMBARK{{{tid},{self.ferry_ids[f]}}}")
        for f, port in enumerate(ferries):
            if active is not None and self.components[port] != active:
                continue
            if f not in self.useful_ferries:
                continue
            for target in self.served[f]:
                # Postpone an empty reposition until a truck waits there.
                # Intermediate empty stops can be replaced by one direct sail.
                if target != port and target not in self.dead_ports and (allowed is not None or occupancy[f] or target in trucks):
                    changed = ferries[:f] + (target,) + ferries[f + 1:]
                    add(f"SAIL{{{self.ferry_ids[f]},{port},{target}}}",
                        (packages, trucks, changed))
        return list(out.values())

    def h_astar(self, node):
        state = node.state
        cached = self._h_cache.get(state)
        if cached is not None:
            self._h_cache.move_to_end(state)
        value = cached if cached is not None else self._lower_bound(state)
        # Pathmax uses only the already supplied parent, never extra search:
        # h(parent) <= cost(parent, child) + h*(child). This preserves stronger
        # bounds when a relaxed packing/route bound drops inconsistently.
        parent = getattr(node, "parent", None)
        if parent is not None:
            previous = getattr(parent, "_logistics_h", 0)
            value = max(value, previous - (node.path_cost - parent.path_cost))
        node._logistics_h = value
        if cached is None and len(self._h_cache) >= 500000:
            self._h_cache.popitem(last=False)
        self._h_cache[state] = value
        return value

    def _drive_forest(self, required, starts, non_sail=False, beat=None):
        """MST with a free super-root joining all current truck positions.

        Each real truck walk visits its required locations. Shortcutting those
        walks in the drive-only metric produces a connected spanning graph
        after joining their starts to the root. In the drive-only metric all
        ferry actions are free, giving a DRIVE bound. The non-sailing metric
        also counts BOARD/DISEMBARK, giving a bound on DRIVE+BOARD+DISEMBARK.
        SAIL is free in both metrics, so shared sailing is never counted here.
        """
        distance = self.non_sail_distance if non_sail else self.drive_distance
        key = (non_sail, frozenset(required), frozenset(starts))
        cached = self._forest_cache.get(key)
        if cached is not None:
            self._forest_cache.move_to_end(key)
            return cached
        pending = {v: min(distance[s][v] for s in starts) for v in required}
        # Connecting each required visit directly to a starting truck is an
        # upper bound on this MST. Skip Prim if it cannot improve the existing
        # lower bound; a skipped result must not be memoized as the true MST.
        if beat is not None and sum(pending.values()) <= beat:
            return 0
        total = 0
        while pending:
            closest = min(pending, key=pending.get)
            cost = pending.pop(closest)
            total += cost
            row = distance[closest]
            for v, old in pending.items():
                if row[v] < old:
                    pending[v] = row[v]
        if len(self._forest_cache) >= 8192:
            self._forest_cache.popitem(last=False)
        self._forest_cache[key] = total
        return total

    def _lower_bound(self, state):
        """Handling + max(individual movement, partitioned weighted workload).

        Admissibility accounting, for an optimal remaining plan:
          B <= its LOAD/UNLOAD cost;
          D <= its DRIVE count; R <= its BOARD/DISEMBARK count;
          S <= its SAIL count; N <= its non-sailing movement cost;
          A <= its complete movement cost.
        Hence B + max(A, max(D + R, N) + 3*S) never exceeds total cost.

        Relaxed shortest routes supply A/N. Weight and exclusive-item workload,
        the rooted drive forest, and mandatory cuts supply action-count bounds.
        Alternatives on the SAME action types are combined with max, not sum.
        Only different road edges or disjoint ferry groups are added in cuts.
        Memoization keys must retain every position/containment fact these
        bounds use; in particular empty trucks aboard ferries are not standing.

        Additional complete-cost bounds use (a) the minimum of transferring a
        parcel for two extra handling actions or keeping its exclusive current
        carrier, and (b) cargo tables charging disjoint parcel handling and road
        sets, plus complementary action costs. They join the final max; they
        are never added to the complete-cost expression above.
        """
        packages, trucks, ferries = state
        vehicle_key = (trucks, ferries)
        vehicle = self._vehicle_cache.get(vehicle_key)
        if vehicle is None:
            vertices = tuple(pos if pos >= 0 else self.aboard_vertex[-1-pos][ferries[-1-pos]]
                             for pos in trucks)
            closest = {}
            vehicle = (vertices, closest)
            if len(self._vehicle_cache) >= 4096:
                self._vehicle_cache.popitem(last=False)
            self._vehicle_cache[vehicle_key] = vehicle
        else:
            self._vehicle_cache.move_to_end(vehicle_key)
            vertices, closest = vehicle
        handling = longest = drives = transfers = sails = 0
        non_sail_longest = 0
        non_sail_work = 0
        trips = {}
        cargo = []
        required = set()
        for p, pos in enumerate(packages):
            goal = self.goals[p]
            if pos == goal:
                continue
            required.add(goal)
            if pos >= 0:
                if not self.capable[p]:
                    # An impossible package supplies no finite useful estimate.
                    # Leave impossibility to the unchanged search algorithm.
                    return 0
                handling += 2
                vertex = pos
                required.add(pos)
                pickup_key = (self.weights[p], pos)
                approach_pair = closest.get(pickup_key)
                if approach_pair is None:
                    approach_pair = (min(self.movement[vertices[t]][pos] for t in self.capable[p]),
                                     min(self.non_sail_distance[vertices[t]][pos] for t in self.capable[p]))
                    closest[pickup_key] = approach_pair
                approach, approach_non_sail = approach_pair
            else:
                handling += 1
                vertex = vertices[-1-pos]
                approach = 0
                approach_non_sail = 0
            travel = self.movement[vertex][goal]
            if travel == float("inf") or approach == float("inf"):
                return 0
            longest = max(longest, approach + travel)
            loc = pos if pos >= 0 else (trucks[-1-pos] if trucks[-1-pos] >= 0
                                        else ferries[-1-trucks[-1-pos]])
            cargo.append((p, loc))
            non_sail_longest = max(non_sail_longest, approach_non_sail + self.non_sail_distance[vertex][goal])
            trips[p] = (self.drive_distance[vertex][goal], self.transfer_distance[vertex][goal],
                        self.sail_distance[vertex][goal], self.non_sail_distance[vertex][goal])
            weight = self.weights[p]
            non_sail_work += weight * trips[p][3]
            drives += weight * self.drive_distance[vertex][goal]
            transfers += weight * self.transfer_distance[vertex][goal]
            sails += weight * self.sail_distance[vertex][goal]
        drive_bound = transfer_bound = sail_bound = 0
        if self.max_capacity > 0:
            drive_bound = (drives + self.max_capacity - 1) // self.max_capacity
            non_sail_longest = max(non_sail_longest,
                                  (non_sail_work + self.max_capacity - 1) // self.max_capacity)
        if required:
            drive_bound = max(drive_bound, self._drive_forest(required, vertices))
        if self.max_capacity > 0:
            transfer_bound = (transfers + self.max_capacity - 1) // self.max_capacity
        if self.sail_capacity > 0:
            sail_bound = (sails + self.sail_capacity - 1) // self.sail_capacity
        # If every selected package weighs > C/(k+1), a truck holds at most k
        # of them. Unit package counts strengthen the weight bound for awkward
        # packing, e.g. weight-2 packages in capacity-3 trucks are all exclusive.
        for group, k in self.dual_groups:
            totals = [0, 0, 0, 0]
            for p in group:
                if p in trips:
                    for metric, distance in enumerate(trips[p]):
                        totals[metric] += distance
            drive_bound = max(drive_bound, (totals[0] + k - 1) // k)
            transfer_bound = max(transfer_bound, (totals[1] + k - 1) // k)
            non_sail_longest = max(non_sail_longest, (totals[3] + k - 1) // k)
            capacity = k * self.max_trucks_per_sail
            if capacity:
                sail_bound = max(sail_bound, (totals[2] + capacity - 1) // capacity)
        edge_drives = ()
        if required:
            cut_drives, cut_transfers, cut_sails, edge_drives = self._cut_bounds(state, required)
            drive_bound = max(drive_bound, cut_drives)
            # Edge counts exclude the residual-flow repair, which may itself
            # traverse non-bridge roads and must not be charged again here.
            drive_bound = max(drive_bound, sum(edge_drives) + self._noncut_cargo_bound(cargo))
            transfer_bound = max(transfer_bound, cut_transfers)
            sail_bound = max(sail_bound, cut_sails)
        if required:
            non_sail_longest = max(non_sail_longest,
                                  self._drive_forest(required, vertices, True,
                                                     max(drive_bound + transfer_bound, non_sail_longest)))
        workload = max(drive_bound + transfer_bound, non_sail_longest) + 3 * sail_bound
        result = handling + max(longest, workload)
        for p, loc in cargo:
            pos = packages[p]
            if pos >= 0:
                continue
            t = -1-pos
            others = tuple((q, place) for q, place in cargo if q != p)
            if not others or any(self.weights[p] + self.weights[q] <= self.capacities[t]
                                 for q, _ in others):
                continue
            # Either keep this parcel in its current small truck (its loaded
            # DRIVEs cannot serve any other remaining parcel), or transfer it
            # with at least two EXTRA handling actions. Min over both cases;
            # other bounds remain valid whichever choice the real plan makes.
            retained = self.drive_distance[vertices[t]][self.goals[p]] + self._cargo_drive_bound(others)
            conditional = min(drive_bound + 2, max(drive_bound, retained))
            result = max(result, handling + conditional + transfer_bound + 3*sail_bound)
        if self.road_cargo_patterns:
            value = sum(table.lookup(tuple(packages[p] for p in pattern), trucks, ferries, mapping)
                        for table, mapping, pattern, _ in self.road_cargo_patterns)
            value += sum(0 if packages[p] == self.goals[p] else 1 if packages[p] < 0 else 2
                         for p in self.pattern_other_packages)
            value += sum(count for edge, count in enumerate(edge_drives) if edge not in self.pattern_edges)
            # Tables own only their parcels' LOAD/UNLOADs and disjoint road
            # sets. Outside cuts, boarding/disembarking and sailing are separate.
            result = max(result, value + transfer_bound + 3*sail_bound)
        return int(result)


def create_logistics_problem(game):
    return LogisticsProblem(game)


if __name__ == "__main__":
    import ex1_check
    ex1_check.main()
