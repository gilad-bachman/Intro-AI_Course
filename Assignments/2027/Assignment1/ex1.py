import heapq
import ex1_check
import search
import utils

id = ["No numbers - I'm special!"]



class LogisticsProblem(search.Problem):
    """This class implements a logistics problem"""

    # Position tags. A position is a tuple (tag, value):
    #   (AT, location)   - standing / lying at a location
    #   (IN, truck_id)   - package inside a truck
    #   (ON, ferry_id)   - truck aboard a ferry
    # Tags are needed because IDs and location numbers may overlap.
    AT = 'L'
    IN = 'T'
    ON = 'F'

    def __init__(self, initial):
        """ Constructor only needs the initial state.
        Don't forget to set the goal or implement the goal test"""

        # ---- Static data (never changes, kept outside the state) ----
        self.num_locations = initial["Locations"]
        self.roads = frozenset(initial["Roads"])
        self.ports = frozenset(initial["Ports"])

        # location -> tuple of locations reachable by one road (roads are two-way)
        neighbors = {l: set() for l in range(self.num_locations)}
        for a, b in self.roads:
            neighbors[a].add(b)
            neighbors[b].add(a)
        self.neighbors = {l: tuple(sorted(ns)) for l, ns in neighbors.items()}

        # truck_id -> max weight capacity
        self.truck_capacity = {t: wmax for t, (_, wmax) in initial["Trucks"].items()}

        # ferry_id -> served ports / max number of trucks
        self.ferry_ports = {f: frozenset(L) for f, (_, L, _) in initial["Ferries"].items()}
        self.ferry_capacity = {f: kmax for f, (_, _, kmax) in initial["Ferries"].items()}

        # package_id -> weight / destination
        self.package_weight = {p: w for p, (_, w, _) in initial["Packages"].items()}
        self.package_goal = {p: goal for p, (_, _, goal) in initial["Packages"].items()}

        # dist[a][b] = cheapest cost between locations a and b (inf if unreachable),
        # a crossing counts BOARD + SAIL + DISEMBARK = 5
        self.dist = self._compute_distances(crossing_cost=5)
        # Same, but a crossing counts only BOARD + DISEMBARK = 2 (no SAIL).
        # Used where costs of several packages are added: one SAIL can carry
        # several trucks, so sails must not be added per package.
        self.dist_no_sail = self._compute_distances(crossing_cost=2)
        # Same, but a crossing costs 0: a lower bound on the DRIVE actions alone
        # (BOARD/DISEMBARK and SAIL are counted by their own bounds)
        self.dist_drive = self._compute_distances(crossing_cost=0)
        # Roads only (no ferries), used to steer empty trucks
        self.road_dist = self._compute_distances(crossing_cost=None)
        # Every port some ferry can dock at
        self.ferry_served_ports = frozenset(
            p for served in self.ferry_ports.values() for p in served)

        # Heaviest load any single truck can carry
        self.max_truck_capacity = max(self.truck_capacity.values(), default=0)

        # Islands and the ferry crossings every route between two islands must use
        self._compute_islands()

        # Roads that cut the map (roads + ferries) in two, used for forced drives
        self._compute_forced_roads()

        # Memoized heuristic values per state (see h_astar)
        self._h_cache = {}

        # Fixed ID order, so every state lists objects in the same order
        self.package_ids = tuple(sorted(initial["Packages"]))
        self.truck_ids = tuple(sorted(initial["Trucks"]))
        self.ferry_ids = tuple(sorted(initial["Ferries"]))

        # ---- Dynamic data (the state) ----
        # state = (packages, trucks, ferries), each a tuple of (id, position)
        # sorted by id. Ferry position is just its docked port.
        packages = tuple((p, (self.AT, initial["Packages"][p][0])) for p in self.package_ids)
        trucks = tuple((t, (self.AT, initial["Trucks"][t][0])) for t in self.truck_ids)
        ferries = tuple((f, initial["Ferries"][f][0]) for f in self.ferry_ids)
        initial_state = (packages, trucks, ferries)

        search.Problem.__init__(self, initial_state)

    def path_cost(self, c, state1, action, state2):
        """ Do not change this. Returns the cost of reaching state2 from state1 via action,
        given cost c to reach state1: SAIL costs 3, every other action costs 1"""
        return c + (3 if action.startswith("SAIL") else 1)

    def successor(self, state):
        """ Generates the successor states returns [(action, achieved_states, ...)]"""
        packages, trucks, ferries = state
        truck_pos = dict(trucks)
        ferry_port = dict(ferries)

        # Pruning: if a package is inside a truck standing at the package's goal,
        # unloading it now is never worse than anything else (any optimal plan
        # can be reordered to do it first, and it only frees capacity), so it is
        # the only successor generated.
        for i, (p, pos) in enumerate(packages):
            if pos[0] == self.IN:
                t = pos[1]
                tpos = truck_pos[t]
                if tpos[0] == self.AT and tpos[1] == self.package_goal[p]:
                    new_packages = self._replace(packages, i, p, tpos)
                    return [("UNLOAD{%d,%d}" % (p, t), (new_packages, trucks, ferries))]

        # Current load of every truck and number of trucks aboard every ferry
        truck_load = {t: 0 for t in self.truck_ids}
        for p, pos in packages:
            if pos[0] == self.IN:
                truck_load[pos[1]] += self.package_weight[p]
        ferry_count = {f: 0 for f in self.ferry_ids}
        for t, pos in trucks:
            if pos[0] == self.ON:
                ferry_count[pos[1]] += 1

        result = []

        # Locations where a truck is standing (not aboard a ferry)
        standing_at = {pos[1] for _, pos in trucks if pos[0] == self.AT}

        # ---- Ferries: SAIL ----
        for i, (f, port) in enumerate(ferries):
            empty_ferry = ferry_count[f] == 0
            for port2 in self.ferry_ports[f]:
                if port2 != port:
                    # Pruning (P1): an empty ferry only sails to a port where a
                    # truck is standing. An empty sail only serves a later BOARD
                    # at its destination; it can be postponed to just before that
                    # BOARD, when the boarding truck stands there (and two empty
                    # sails in a row can be merged into one).
                    if empty_ferry and port2 not in standing_at:
                        continue
                    new_ferries = self._replace(ferries, i, f, port2)
                    result.append(("SAIL{%d,%d,%d}" % (f, port, port2),
                                   (packages, trucks, new_ferries)))

        # ---- Packages: LOAD / UNLOAD ----
        for i, (p, pos) in enumerate(packages):
            if pos[0] == self.AT:
                # Pruning: a package lying at its goal is never loaded again
                if pos[1] == self.package_goal[p]:
                    continue
                # LOAD into any truck standing at the same location with room
                for t in self.truck_ids:
                    if (truck_pos[t] == pos and
                            truck_load[t] + self.package_weight[p] <= self.truck_capacity[t]):
                        new_packages = self._replace(packages, i, p, (self.IN, t))
                        result.append(("LOAD{%d,%d}" % (p, t),
                                       (new_packages, trucks, ferries)))
            else:
                # UNLOAD only if its truck is standing at a location
                t = pos[1]
                tpos = truck_pos[t]
                if tpos[0] == self.AT:
                    new_packages = self._replace(packages, i, p, tpos)
                    result.append(("UNLOAD{%d,%d}" % (p, t),
                                   (new_packages, trucks, ferries)))

        # Locations where something still has to happen: undelivered packages
        # lying there, and goals of undelivered packages
        useful = set()
        pickups = set()     # locations with an undelivered package lying there
        for p, pos in packages:
            goal = self.package_goal[p]
            if pos[0] == self.AT:
                if pos[1] != goal:
                    useful.add(pos[1])
                    useful.add(goal)
                    pickups.add(pos[1])
            else:
                useful.add(goal)
        # Where an empty truck may be heading: a package to pick up, or a port
        empty_targets = pickups | self.ferry_served_ports
        road_dist = self.road_dist

        # ---- Trucks: DRIVE / BOARD / DISEMBARK ----
        cut_far = self.cut_far
        for i, (t, pos) in enumerate(trucks):
            if pos[0] == self.AT:
                l = pos[1]
                # Pruning (P2): an empty truck only drives one step along a
                # shortest road path toward a target (a package to pick up, or a
                # port). An empty truck drives only to LOAD or BOARD somewhere;
                # its drives can be postponed to just before that action and
                # replaced by a shortest road path, which is never more expensive.
                if truck_load[t] == 0:
                    here = road_dist[l]
                    toward = [x for x in empty_targets if here[x] != float('inf')]
                else:
                    toward = None
                for l2 in self.neighbors[l]:
                    if toward is not None:
                        there = road_dist[l2]
                        if not any(there[x] == here[x] - 1 for x in toward):
                            continue
                    # Pruning: never drive across a cut road into a region where
                    # nothing has to happen (no package to pick up, no goal).
                    # Anything a truck could do in there (e.g. drop a package off
                    # for another truck) can be done at the entrance instead.
                    far = cut_far.get((l, l2))
                    if far is not None and far.isdisjoint(useful):
                        continue
                    new_trucks = self._replace(trucks, i, t, (self.AT, l2))
                    result.append(("DRIVE{%d,%d,%d}" % (t, l, l2),
                                   (packages, new_trucks, ferries)))
                # BOARD any ferry docked here that still has room
                for f in self.ferry_ids:
                    if ferry_port[f] == l and ferry_count[f] < self.ferry_capacity[f]:
                        new_trucks = self._replace(trucks, i, t, (self.ON, f))
                        result.append(("BOARD{%d,%d}" % (t, f),
                                       (packages, new_trucks, ferries)))
            else:
                # DISEMBARK at the port where its ferry is docked
                f = pos[1]
                new_trucks = self._replace(trucks, i, t, (self.AT, ferry_port[f]))
                result.append(("DISEMBARK{%d,%d}" % (t, f),
                               (packages, new_trucks, ferries)))

        return result

    def _compute_distances(self, crossing_cost):
        """All-pairs shortest distances on a graph of the locations:
        every road is an edge of cost 1, and every pair of ports served by the
        same ferry is an edge of cost crossing_cost (no ferry edges if
        crossing_cost is None). Dijkstra from every location."""
        n = self.num_locations
        edges = {l: {} for l in range(n)}
        for a, b in self.roads:
            edges[a][b] = min(edges[a].get(b, 1), 1)
            edges[b][a] = min(edges[b].get(a, 1), 1)
        for f, served in self.ferry_ports.items():
            if crossing_cost is None:
                break
            for a in served:
                for b in served:
                    if a != b:
                        edges[a][b] = min(edges[a].get(b, crossing_cost), crossing_cost)

        INF = float('inf')
        dist = []
        for src in range(n):
            d = [INF] * n
            d[src] = 0
            heap = [(0, src)]
            while heap:
                cost, u = heapq.heappop(heap)
                if cost > d[u]:
                    continue
                for v, w in edges[u].items():
                    if cost + w < d[v]:
                        d[v] = cost + w
                        heapq.heappush(heap, (cost + w, v))
            dist.append(d)
        return dist

    def _compute_islands(self):
        """Static island structure, used by the ferry part of the heuristic.
        self.island[l]          - island (road-connected component) of location l
        self.crossings          - directed island crossings (X, Y), X != Y, that
                                  some ferry can sail
        self.forced[(I, J)]     - crossings that EVERY island route from I to J uses
        self.crossing_ferries   - (X, Y) -> ferries that can sail from X to Y"""
        n = self.num_locations
        island = [-1] * n
        k = 0
        for start in range(n):
            if island[start] != -1:
                continue
            island[start] = k
            stack = [start]
            while stack:
                u = stack.pop()
                for v in self.neighbors[u]:
                    if island[v] == -1:
                        island[v] = k
                        stack.append(v)
            k += 1
        self.island = island
        self.num_islands = k

        crossing_ferries = {}
        for f, served in self.ferry_ports.items():
            for a in served:
                for b in served:
                    if island[a] != island[b]:
                        crossing_ferries.setdefault((island[a], island[b]), set()).add(f)
        self.crossing_ferries = {c: frozenset(fs) for c, fs in crossing_ferries.items()}
        self.crossings = tuple(sorted(self.crossing_ferries))
        # Most weight one SAIL can carry across each crossing: the biggest ferry
        # that can make it, filled with trucks of the biggest capacity
        self.crossing_capacity = {
            c: max(self.ferry_capacity[f] for f in fs) * self.max_truck_capacity
            for c, fs in self.crossing_ferries.items()}
        # Most weight any single SAIL can carry anywhere
        self.max_sail_capacity = max(self.crossing_capacity.values(), default=0)

        # Group crossings that share a ferry (transitively). One ride (BOARD ...
        # DISEMBARK) is on one ferry, so it can only cover crossings of one group.
        parent = {c: c for c in self.crossings}

        def find(c):
            while parent[c] != c:
                parent[c] = parent[parent[c]]
                c = parent[c]
            return c

        for f in self.ferry_ports:
            served = [c for c, fs in self.crossing_ferries.items() if f in fs]
            for c in served[1:]:
                parent[find(c)] = find(served[0])
        self.crossing_group = {c: find(c) for c in self.crossings}

        # island_hops[i][j] = fewest crossings (SAILs) from island i to island j
        INF = float('inf')
        hops = [[INF] * k for _ in range(k)]
        for i in range(k):
            hops[i][i] = 0
            frontier = [i]
            while frontier:
                nxt = []
                for u in frontier:
                    for (x, y) in self.crossings:
                        if x == u and hops[i][y] == INF:
                            hops[i][y] = hops[i][u] + 1
                            nxt.append(y)
                frontier = nxt
        self.island_hops = hops

        # island_rides[i][j] = fewest ferry RIDES from island i to island j. A
        # ride is one truck from BOARD to DISEMBARK on one ferry: it can stay
        # aboard for several sails, but a ferry sails directly between any two
        # of its ports, so one ride reaches every island that ferry serves. A new
        # ride (a new BOARD + DISEMBARK) is needed only to change ferries.
        ferry_islands = [frozenset(island[p] for p in served)
                         for served in self.ferry_ports.values()]
        one_ride = [set() for _ in range(k)]        # islands one ride away
        for isl in ferry_islands:
            for a in isl:
                one_ride[a] |= isl - {a}
        rides = [[INF] * k for _ in range(k)]
        for i in range(k):
            rides[i][i] = 0
            frontier = [i]
            while frontier:
                nxt = []
                for u in frontier:
                    for v in one_ride[u]:
                        if rides[i][v] == INF:
                            rides[i][v] = rides[i][u] + 1
                            nxt.append(v)
                frontier = nxt
        self.island_rides = rides

        def reachable(src, banned):
            seen = {src}
            stack = [src]
            while stack:
                u = stack.pop()
                for (x, y) in self.crossings:
                    if x == u and (x, y) != banned and y not in seen:
                        seen.add(y)
                        stack.append(y)
            return seen

        reach_all = [reachable(i, None) for i in range(k)]
        forced = {}
        for c in self.crossings:
            reach_without = [reachable(i, c) for i in range(k)]
            for i in range(k):
                for j in reach_all[i]:
                    if j != i and j not in reach_without[i]:
                        forced.setdefault((i, j), []).append(c)
        self.forced = {pair: tuple(cs) for pair, cs in forced.items()}

    def _compute_forced_roads(self):
        """A road is a cut road if removing it disconnects its two ends in the
        full map (roads + every ferry connection). For cut road i:
        self.cut_side[i][l] = 0 if l is on the side of its first end,
                              1 if on the side of its second end,
                             -1 if l is not connected to that road at all.
        Every route between two locations on different sides must drive the
        road, in the direction from the start's side to the target's side."""
        n = self.num_locations
        adj = {l: [] for l in range(n)}         # (neighbor, edge id); ferries id -1
        road_list = sorted(tuple(r) for r in self.roads)
        for i, (a, b) in enumerate(road_list):
            adj[a].append((b, i))
            adj[b].append((a, i))
        for served in self.ferry_ports.values():
            for a in served:
                for b in served:
                    if a != b:
                        adj[a].append((b, -1))

        def component(src, banned):
            seen = {src}
            stack = [src]
            while stack:
                u = stack.pop()
                for v, e in adj[u]:
                    if e != banned and v not in seen:
                        seen.add(v)
                        stack.append(v)
            return seen

        self.cut_roads = []         # (first end, second end) per cut road
        self.cut_side = []
        for i, (a, b) in enumerate(road_list):
            side_a = component(a, i)
            if b in side_a:
                continue                         # another route exists
            side_b = component(b, i)
            side = [-1] * n
            for l in side_a:
                side[l] = 0
            for l in side_b:
                side[l] = 1
            self.cut_roads.append((a, b))
            self.cut_side.append(side)
        # Lazily filled caches of _forced_drives / _forced_drives_set results,
        # indexed [src][dst]
        self._forced_list = [[None] * n for _ in range(n)]
        self._forced_set = [[None] * n for _ in range(n)]
        # (truck locations, package location) -> directions every truck must
        # drive to reach the package (see h_astar); cleared if it grows too big
        self._reach_cache = {}

        # (u, v) -> locations on v's side of cut road u-v: the region a truck
        # enters by driving u -> v (it can only enter and leave through u-v)
        self.cut_far = {}
        for (a, b), side in zip(self.cut_roads, self.cut_side):
            self.cut_far[(a, b)] = frozenset(l for l in range(n) if side[l] == 1)
            self.cut_far[(b, a)] = frozenset(l for l in range(n) if side[l] == 0)

    def _forced_drives(self, src, dst):
        """Directed cut roads every route from src to dst must drive, as a tuple
        of ints. A direction is encoded as 2 * (cut road index) + s, where s is
        the side it starts from (0 = first end -> second end, 1 = the reverse)."""
        res = self._forced_list[src][dst]
        if res is None:
            out = []
            for i, side in enumerate(self.cut_side):
                s, d = side[src], side[dst]
                if s != -1 and d != -1 and s != d:
                    out.append(2 * i + s)        # from side s toward the other
            res = tuple(out)
            self._forced_list[src][dst] = res
        return res

    def _forced_drives_set(self, src, dst):
        """Same as _forced_drives, as a frozenset (for intersections)."""
        res = self._forced_set[src][dst]
        if res is None:
            res = frozenset(self._forced_drives(src, dst))
            self._forced_set[src][dst] = res
        return res

    def _reach_forced(self, truck_locs, loc):
        """Directions EVERY truck must drive to reach loc: the intersection over
        the trucks' locations of their forced directions toward loc. Cached by
        (truck locations, loc), since most moves leave both unchanged."""
        key = (truck_locs, loc)
        res = self._reach_cache.get(key)
        if res is None:
            inter = None
            for tl in truck_locs:
                f = self._forced_drives_set(tl, loc)
                inter = f if inter is None else (inter & f)
                if not inter:
                    break
            res = tuple(inter) if inter else ()
            if len(self._reach_cache) > 500000:
                self._reach_cache.clear()
            self._reach_cache[key] = res
        return res

    def _mst_bound(self, points, start_cost, beat=-1):
        """Minimum spanning tree over the trucks and the points that some truck
        must still visit (Prim's algorithm).

        Why it is a lower bound on the DRIVE/BOARD/DISEMBARK actions left:
        list the points each truck visits in the order it first reaches them.
        Every leg between two consecutive points costs at least their no-sail
        distance, and the legs of one truck form a path starting at that truck.
        All trucks' paths together connect every point to some truck, i.e. they
        form a spanning tree once all trucks are joined by 0-cost edges. The MST
        is the cheapest such tree, so it costs at most what the trucks drive.

        points     - set of locations to connect
        start_cost - start_cost(x): cheapest no-sail cost for any truck to reach
                     x; all trucks are in the tree from the start, so this is the
                     initial key of every point.
        beat       - only the result matters if it is larger than this. Joining
                     every point straight to its closest truck is one spanning
                     tree, so the sum of the initial keys is an upper bound on
                     the MST; if that is <= beat, the MST cannot win and 0 is
                     returned without building it."""
        if not points:
            return 0
        dist_ns = self.dist_no_sail
        key = {x: start_cost(x) for x in points}
        if sum(key.values()) <= beat:
            return 0
        total = 0
        while key:
            x = min(key, key=key.get)
            total += key.pop(x)
            row = dist_ns[x]
            for y in key:
                d = row[y]
                if d < key[y]:
                    key[y] = d
        return total

    def _sail_lower_bound(self, remaining_islands, ferry_port):
        """Returns (lower bound on SAIL actions still needed, lower bound on truck
        rides still needed).
        Rides: every package that changes island rides in trucks on ferries, at
        least island_rides(i, j) times, and one ride carries at most C_max
        weight, so rides >= ceil(sum(weight * rides) / C_max). The forced-crossing
        ride count (below) is another lower bound; the larger one is returned.
        Lower bound on the number of SAIL actions:
        remaining_islands - list of (current island, goal island, weight) of the
                            undelivered packages.
        1. Every crossing forced for some packages needs SAILs of its own, and
           different crossings are different SAIL actions. One SAIL carries at
           most crossing_capacity weight, so a crossing that W weight is forced
           through needs at least ceil(W / crossing_capacity) SAILs.
        2. If a forced crossing X->Y cannot start now (no ferry able to sail
           X->Y is docked on X) and no forced crossing ends in X, some extra SAIL
           must bring a ferry into X first. Extra sails into different islands
           are different actions, and none of them is a forced crossing."""
        # Bound 1 (no forced crossings needed): each package must be carried on
        # at least hops(i, j) SAILs, so sum(weight * hops) is the weight-sails
        # still to do, and one SAIL does at most max_sail_capacity of them.
        workload = 0
        ride_workload = 0
        for i, j, w in remaining_islands:
            if i != j:
                workload += w * self.island_hops[i][j]
                ride_workload += w * self.island_rides[i][j]
        cap = self.max_sail_capacity
        by_workload = -(-workload // cap) if cap > 0 and workload else 0
        c_max = self.max_truck_capacity
        rides_by_workload = (-(-ride_workload // c_max)
                             if c_max > 0 and ride_workload else 0)

        # Bound 2: forced crossings (below). Both bound the total number of
        # SAILs, so the answer is the larger of the two.
        needed = {}                         # crossing -> weight forced through it
        for i, j, w in remaining_islands:
            if i != j:
                for c in self.forced.get((i, j), ()):
                    needed[c] = needed.get(c, 0) + w
        if not needed:
            return by_workload, rides_by_workload

        # Truck rides: a forced crossing with weight W needs ceil(W / C_max) rides.
        # Crossings in one group may share rides (take the max); different
        # groups never do (add them up).
        group_rides = {}
        for c, weight in needed.items():
            r = -(-weight // c_max) if c_max > 0 else 1
            g = self.crossing_group[c]
            if r > group_rides.get(g, 0):
                group_rides[g] = r
        rides = sum(group_rides.values())
        if rides_by_workload > rides:
            rides = rides_by_workload
        sails = 0
        for c, weight in needed.items():
            per_sail = self.crossing_capacity[c]
            trips = -(-weight // per_sail) if per_sail > 0 else 1
            sails += trips if trips > 1 else 1
        island = self.island
        arrive = {y for (_, y) in needed}
        late = set()
        for (x, y) in needed:
            if x in arrive or x in late:
                continue
            if not any(island[ferry_port[f]] == x for f in self.crossing_ferries[(x, y)]):
                late.add(x)
        by_forced = sails + len(late)
        return (by_forced if by_forced > by_workload else by_workload), rides

    @staticmethod
    def _replace(entries, i, obj_id, new_pos):
        """Return a copy of the (id, position) tuple with entry i set to new_pos"""
        return entries[:i] + ((obj_id, new_pos),) + entries[i + 1:]

    def goal_test(self, state):
        """ given a state, checks if this is the goal state, compares to the created goal state returns True/False"""
        packages = state[0]
        for p, pos in packages:
            if pos != (self.AT, self.package_goal[p]):
                return False
        return True

    def h_astar(self, node):
        """ This is the heuristic. It gets a node (not a state)
        and returns a goal distance estimate.

        The same state is often generated many times (from different parents),
        so the value is memoized per state. This only avoids recomputing the
        same number: the value for every state is exactly _h_value(state), and
        no states are explored here (only states the search generated)."""
        state = node.state
        h = self._h_cache.get(state)
        if h is None:
            h = self._h_value(state)
            if len(self._h_cache) > 3000000:     # bound memory use
                self._h_cache.clear()
            self._h_cache[state] = h
        return h

    def _h_value(self, state):
        """The heuristic value of a state (see h_astar)."""
        packages, trucks, ferries = state
        ferry_port = dict(ferries)
        dist = self.dist
        dist_ns = self.dist_no_sail

        # Where every truck is: (location, ferry it is aboard or None).
        # A truck aboard a ferry is at the ferry's port.
        truck_at = {}
        for t, pos in trucks:
            if pos[0] == self.AT:
                truck_at[t] = (pos[1], None)
            else:
                truck_at[t] = (ferry_port[pos[1]], pos[1])

        def truck_dist(t, target, table, sail_cost):
            """Lower bound on the cost for truck t to stand at target.
            A truck aboard a ferry must first (optionally sail, then) DISEMBARK
            at one of the ferry's ports, and continue from there."""
            loc, f = truck_at[t]
            if f is None:
                return table[loc][target]
            best = 1 + table[loc][target]                      # DISEMBARK here
            for port in self.ferry_ports[f]:
                if port != loc:
                    c = sail_cost + 1 + table[port][target]     # SAIL, DISEMBARK
                    if c < best:
                        best = c
            return best

        # Packages not yet delivered: (package, weight, its position)
        remaining = []
        for p, pos in packages:
            if pos[0] == self.AT and pos[1] == self.package_goal[p]:
                continue
            remaining.append((p, self.package_weight[p], pos))
        if not remaining:
            return 0

        # Two smallest remaining weights, to decide quickly whether a package can
        # share a truck with any other remaining package
        weights = sorted(w for _, w, _ in remaining)
        min_w = weights[0]
        second_w = weights[1] if len(weights) > 1 else None
        cap = self.max_truck_capacity

        island = self.island
        load_unload = 0     # sum over packages: LOAD/UNLOAD actions still needed
        # Movement terms without any SAIL (sails are counted separately)
        movement_ns = 0     # max over packages: truck -> package -> goal
        shared_max_ns = 0   # max over non-solo packages: carried trip
        solo_sum = 0        # sum over solo packages: carried trip
        # Same two, counting DRIVE actions only (crossings free)
        dist_drive = self.dist_drive
        shared_max_drive = 0
        solo_sum_drive = 0
        islands = []        # (current island, goal island) per remaining package
        # directed cut road -> total weight of packages forced through it.
        # A direction with weight 0 is one some truck must drive just to reach
        # a package.
        forced = {}
        must_visit = set()  # locations some truck must still stand at
        truck_locs = frozenset(truck_at[t][0] for t in self.truck_ids)
        forced_get = forced.get
        for p, w, pos in remaining:
            goal = self.package_goal[p]
            if pos[0] == self.AT:
                loc = pos[1]
                load_unload += 2                                # LOAD + UNLOAD
                reach_ns = min(truck_dist(t, loc, dist_ns, 0) for t in self.truck_ids)
                carried_ns = dist_ns[loc][goal]
                carried_drive = dist_drive[loc][goal]
                need_ns = reach_ns + carried_ns                 # truck -> package -> goal
                here = island[loc]
                # Cut roads EVERY truck must cross to reach this package: some
                # truck has to drive them toward the package.
                for d in self._reach_forced(truck_locs, loc):
                    if d not in forced:
                        forced[d] = 0
            else:
                t = pos[1]
                load_unload += 1                                # UNLOAD
                carried_ns = truck_dist(t, goal, dist_ns, 0)
                # Drives only: a truck aboard a ferry is at the ferry's port and
                # crossings are free in dist_drive, so no DISEMBARK is added here
                # (it is counted in board_disembark)
                carried_drive = dist_drive[truck_at[t][0]][goal]
                need_ns = carried_ns                            # its truck -> goal
                loc = truck_at[t][0]
                here = island[loc]
            # Cut roads every route of this package to its goal uses: the
            # package's weight must be carried across each of them
            for d in self._forced_drives(loc, goal):
                forced[d] = forced_get(d, 0) + w
            # A truck must stand at the goal to UNLOAD, and at a lying package's
            # location to LOAD it
            must_visit.add(goal)
            if pos[0] == self.AT:
                must_visit.add(loc)
            if need_ns > movement_ns:
                movement_ns = need_ns
            islands.append((here, island[goal], w))

            # Solo: cannot share even the biggest truck with any other remaining
            # package. While it is carried its truck holds nothing else, so the
            # DRIVE / BOARD / DISEMBARK actions of its carried trip serve only it.
            other_min = second_w if w == min_w else min_w
            if other_min is not None and w + other_min > cap:
                solo_sum += carried_ns
                solo_sum_drive += carried_drive
            else:
                if carried_ns > shared_max_ns:
                    shared_max_ns = carried_ns
                if carried_drive > shared_max_drive:
                    shared_max_drive = carried_drive

        # h = LOAD/UNLOAD + DRIVE/BOARD/DISEMBARK (no sails) + SAILs.
        # The three parts count different action types, so they can be added.
        # The DRIVE/BOARD/DISEMBARK part is the max of several lower bounds.
        # Carried trips of different solo packages, and of one non-solo package,
        # never share a DRIVE/BOARD/DISEMBARK, so they can be added.
        truck_ids = self.truck_ids
        # Forced drives: one DRIVE across a road carries at most the biggest
        # truck capacity, so a direction that W weight must cross needs at least
        # ceil(W / cap) DRIVE actions (at least 1 if a truck must use it at all).
        # Different directions are different actions, so the counts add up.
        forced_drives = 0
        if cap > 0:
            for weight in forced.values():
                trips = -(-weight // cap)
                forced_drives += trips if trips > 1 else 1
        else:
            forced_drives = len(forced)
        sails, rides = self._sail_lower_bound(islands, ferry_port)
        # Every ride ends with a DISEMBARK and starts with a BOARD, except rides
        # already under way (at most one per truck already aboard a ferry).
        aboard = sum(1 for t in truck_ids if truck_at[t][1] is not None)
        board_disembark = 2 * rides - aboard
        if board_disembark < 0:
            board_disembark = 0
        # forced_drives (DRIVE) + board_disembark (BOARD/DISEMBARK) count
        # different actions, so together they are one more movement candidate.
        # DRIVE-only bounds: forced drives, and the drives of carried trips that
        # cannot share a DRIVE (solo packages + one non-solo). Both count only
        # DRIVE actions, so take the max; then add BOARD/DISEMBARK, a different
        # action type.
        drive_bound = max(forced_drives, solo_sum_drive + shared_max_drive)
        movement_ns = max(movement_ns, solo_sum + shared_max_ns,
                          drive_bound + board_disembark)
        # MST over trucks + places to visit: also a bound on DRIVE/BOARD/DISEMBARK
        # (no sails), so it joins the same max. It is only built when it could
        # beat the current best (see _mst_bound).
        mst = self._mst_bound(
            must_visit,
            lambda x: min(truck_dist(t, x, dist_ns, 0) for t in truck_ids),
            movement_ns)
        if mst > movement_ns:
            movement_ns = mst

        return load_unload + movement_ns + 3 * sails


def create_logistics_problem(game):
    print("<<create_logistics_problem")
    """ Create a logistics problem, based on the description.
    game - dictionary as described in pdf file"""
    return LogisticsProblem(game)


if __name__ == '__main__':
    ex1_check.main()
