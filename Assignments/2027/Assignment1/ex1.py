# AI/source disclosure: developed with Claude and OpenAI Codex for state design,
# admissible heuristics, pruning, code implementation, and testing.
# Online ideas consulted: Wehrle et al., IJCAI 2015 (symmetry/partial-order
# reduction), https://ijcai.org/proceedings/2015/244; Paul et al., SOCS 2017,
# https://ojs.aaai.org/index.php/SOCS/article/download/18428/18219/21944.
# Implementations below are original adaptations; capacities and fixed A* remain
# governed by Assignment1.pdf. No stored plans or problem-name detection.

import heapq
import ex1_check
import search
import utils

id = ["No numbers - I'm special!"]



class _MapProblem(search.Problem):
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

        # Components of the FULL transport map, including ferry links. Actions
        # in different components affect disjoint objects and commute.
        self.transport_component = [-1] * self.num_locations
        component = 0
        for src in range(self.num_locations):
            if self.transport_component[src] != -1:
                continue
            for dst, distance in enumerate(self.dist[src]):
                if distance != float('inf'):
                    self.transport_component[dst] = component
            component += 1
        self.num_transport_components = component

        # Memoized heuristic values per state (see h_astar)
        self._h_cache = {}

        # Fixed ID order, so every state lists objects in the same order
        self.package_ids = tuple(sorted(initial["Packages"]))
        self.truck_ids = tuple(sorted(initial["Trucks"]))
        self.ferry_ids = tuple(sorted(initial["Ferries"]))
        self._compute_packing_bounds()

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
        # Bit d says a truck at this location can start a drive in cut direction
        # d. Combining trucks' masks answers the common "any truck on this
        # side?" test without scanning all trucks for every cut road.
        self._cut_direction_bits = tuple(1 << d for d in range(2 * len(self.cut_side)))
        self._cut_even_mask = sum(self._cut_direction_bits[::2])
        self._truck_cut_masks = []
        for loc in range(n):
            mask = 0
            for road, side in enumerate(self.cut_side):
                if side[loc] != -1:
                    mask |= self._cut_direction_bits[2 * road + side[loc]]
            self._truck_cut_masks.append(mask)

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

    def _compute_packing_bounds(self):
        """Minimum loads for subsets in a relaxed fleet of C_max trucks.

        This is subset dynamic programming on package weights, not a search
        over logistics states. For each subset retain the fewest loads and,
        among ties, the lightest last load. Adding its final package either
        fits in that load or opens one new load. Unlimited copies of the
        largest truck only make the real problem easier.

        Bound the exponential precomputation to 16 packages. Larger problems
        retain the existing admissible weight bounds. If every subset already
        meets the weight-division bound, disable runtime packing work entirely.
        """
        self._packing_loads = None
        self._package_bits = {}
        cap = self.max_truck_capacity
        weights = [self.package_weight[p] for p in self.package_ids]
        n = len(weights)
        if (not n or n > 16 or cap <= 0 or max(weights) > cap or
                sum(weights) <= cap or
                (len(set(weights)) == 1 and weights[0] > 0 and cap % weights[0] == 0)):
            return
        size = 1 << n
        loads = [0] * size
        last = [0] * size
        total = [0] * size
        improves = False
        for subset in range(1, size):
            bit = subset & -subset
            total[subset] = total[subset ^ bit] + weights[bit.bit_length() - 1]
            best_count, best_last = n + 1, 0
            pending = subset
            while pending:
                bit = pending & -pending
                pending ^= bit
                previous = subset ^ bit
                weight = weights[bit.bit_length() - 1]
                count, used = loads[previous], last[previous]
                if count == 0 or used + weight > cap:
                    count += 1
                    used = weight
                else:
                    used += weight
                if count < best_count or (count == best_count and used < best_last):
                    best_count, best_last = count, used
            loads[subset], last[subset] = best_count, best_last
            if best_count > -(-total[subset] // cap):
                improves = True
        if improves:
            self._packing_loads = loads
            self._package_bits = {p: 1 << i for i, p in enumerate(self.package_ids)}

    def _sail_lower_bound(self, remaining_islands, ferry_port, package_masks=None):
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
        # Two packages heavier than half a container cannot share it. Their
        # required sails/rides therefore count separately, without choosing a
        # particular route. This supplements the fractional weight bound.
        heavy_sails = 0
        heavy_rides = 0
        for i, j, w in remaining_islands:
            if i != j:
                workload += w * self.island_hops[i][j]
                ride_workload += w * self.island_rides[i][j]
                if 2 * w > self.max_sail_capacity:
                    heavy_sails += self.island_hops[i][j]
                if 2 * w > self.max_truck_capacity:
                    heavy_rides += self.island_rides[i][j]
        cap = self.max_sail_capacity
        by_workload = -(-workload // cap) if cap > 0 and workload else 0
        c_max = self.max_truck_capacity
        rides_by_workload = (-(-ride_workload // c_max)
                             if c_max > 0 and ride_workload else 0)
        by_workload = max(by_workload, heavy_sails)
        rides_by_workload = max(rides_by_workload, heavy_rides)

        # Bound 2: forced crossings (below). Both bound the total number of
        # SAILs, so the answer is the larger of the two.
        needed = {}                         # crossing -> weight forced through it
        packing = self._packing_loads if package_masks is not None else None
        for index, (i, j, w) in enumerate(remaining_islands):
            if i != j:
                for c in self.forced.get((i, j), ()):
                    if packing is None:
                        needed[c] = needed.get(c, 0) + w
                    else:
                        needed[c] = needed.get(c, 0) | package_masks[index]
        if not needed:
            return by_workload, rides_by_workload

        # Truck rides: a forced crossing with weight W needs ceil(W / C_max) rides.
        # Crossings in one group may share rides (take the max); different
        # groups never do (add them up).
        group_rides = {}
        for c, weight in needed.items():
            r = (packing[weight] if packing is not None else
                 (-(-weight // c_max) if c_max > 0 else 1))
            g = self.crossing_group[c]
            if r > group_rides.get(g, 0):
                group_rides[g] = r
        rides = sum(group_rides.values())
        if rides_by_workload > rides:
            rides = rides_by_workload
        sails = 0
        for c, weight in needed.items():
            per_sail = self.crossing_capacity[c]
            if packing is not None and per_sail > 0:
                trucks_per_sail = per_sail // c_max
                trips = -(-packing[weight] // trucks_per_sail)
            else:
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
        packing = self._packing_loads
        package_masks = [] if packing is not None else None
        # Directed cut road -> package subset if packing helps this instance,
        # otherwise total weight. Zero also marks pickup-only directions.
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
            if packing is None:
                for d in self._forced_drives(loc, goal):
                    forced[d] = forced_get(d, 0) + w
            else:
                bit = self._package_bits[p]
                for d in self._forced_drives(loc, goal):
                    forced[d] = forced_get(d, 0) | bit
                package_masks.append(bit)
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
        # With at most one trip per direction in a connected map, returns
        # cannot strengthen the existing bound: a package on a side with no
        # truck already forces a pickup drive in the reverse direction. Skip
        # the extra accounting in that case (notably when all packages fit in
        # one truck, as on x4).
        packed_trips = ({d: max(1, packing[subset]) for d, subset in forced.items()}
                        if packing is not None else None)
        single_trip = (max(packed_trips.values(), default=0) <= 1
                       if packed_trips is not None else
                       cap > 0 and max(forced.values(), default=0) <= cap)
        if self.num_transport_components == 1 and single_trip:
            forced_drives = len(forced)
        else:
            crossing_trips = {}
            forced_drives = required_mask = repeated_mask = 0
            repeated = []
            bits = self._cut_direction_bits
            for direction, weight in forced.items():
                trips = (packed_trips[direction] if packed_trips is not None else
                         (-(-weight // cap) if cap > 0 else 1))
                if trips < 1:
                    trips = 1
                crossing_trips[direction] = trips
                forced_drives += trips
                required_mask |= bits[direction]
                if trips > 1:
                    repeated.append((direction, trips))
                    repeated_mask |= bits[direction]
            # Conservation across a cut road: if a departures from side 0 are
            # required and only T0 trucks are there now, at least a-T0 reverse
            # drives must supply the remaining trucks. The reverse deliveries can
            # do that job, so use max, never add the two requirements. Ferries
            # cannot bypass a cut road (cuts were computed on the full map).
            occupied_sides = 0
            if crossing_trips:
                for location, _ in truck_at.values():
                    occupied_sides |= self._truck_cut_masks[location]
            # Swap adjacent bits to identify directions with reverse deliveries.
            # For a direction used once, a return is forced exactly when neither a
            # truck nor a required reverse crossing can supply that departure.
            reverse_mask = ((required_mask & self._cut_even_mask) << 1 |
                            (required_mask & ~self._cut_even_mask) >> 1)
            forced_drives += (required_mask & ~repeated_mask &
                              ~reverse_mask & ~occupied_sides).bit_count()
            # Only directions requiring several trips need actual truck counts.
            for direction, departures in repeated:
                shortage = departures - crossing_trips.get(direction ^ 1, 0)
                if shortage <= 0:
                    continue
                departure_side = direction & 1
                side = self.cut_side[direction // 2]
                for location, _ in truck_at.values():
                    if side[location] == departure_side:
                        shortage -= 1
                        if shortage == 0:
                            break
                if shortage > 0:
                    forced_drives += shortage
        sails, rides = self._sail_lower_bound(islands, ferry_port, package_masks)
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



class State:
    __slots__=('data','key','aliases','_hash')
    def __init__(self,data,key,aliases):
        object.__setattr__(self,'data',data)
        object.__setattr__(self,'key',key)
        object.__setattr__(self,'aliases',aliases)
        object.__setattr__(self,'_hash',hash(key))
    def __setattr__(self,k,v):raise AttributeError('immutable state')
    def __hash__(self):return self._hash
    def __eq__(self,other):return isinstance(other,State) and self.key==other.key

class _CompactProblem(_MapProblem):
    def __init__(self,game):
        super().__init__(game)
        self.P=len(self.package_ids);self.T=len(self.truck_ids);self.F=len(self.ferry_ids);self.N=self.num_locations
        self.ws=tuple(self.package_weight[p] for p in self.package_ids)
        self.gs=tuple(self.package_goal[p] for p in self.package_ids)
        self.cs=tuple(self.truck_capacity[t] for t in self.truck_ids)
        self.ks=tuple(self.ferry_capacity[f] for f in self.ferry_ids)
        self.fs=tuple(tuple(sorted(self.ferry_ports[f])) for f in self.ferry_ids)
        def groups(values):
            d={}
            for i,value in enumerate(values):d.setdefault(value,[]).append(i)
            return tuple(tuple(g) for g in d.values())
        self.pgroups=groups(tuple(zip(self.ws,self.gs)))
        self.tgroups=groups(self.cs)
        self.fgroups=groups(tuple(zip(self.fs,self.ks)))
        self.pclass=tuple(next(j for j,g in enumerate(self.pgroups) if i in g) for i in range(self.P))
        self.fclass=tuple(next(j for j,g in enumerate(self.fgroups) if i in g) for i in range(self.F))
        self._tree_setup()
        # Peel transport-map dead ends with no initial truck/package or goal.
        adj=[set(self.neighbors[v]) for v in range(self.N)]
        for served in self.fs:
            for v in served:adj[v].update(w for w in served if w!=v)
        terminals={v[0] for v in game['Trucks'].values()}|{v[0] for v in game['Packages'].values()}|set(self.gs)
        pending=[v for v in range(self.N) if len(adj[v])<=1 and v not in terminals]
        removed=set()
        while pending:
            v=pending.pop()
            if v in removed:continue
            removed.add(v)
            for w in list(adj[v]):
                adj[w].discard(v)
                if len(adj[w])<=1 and w not in terminals:pending.append(w)
            adj[v].clear()
        self.dead_locations=removed
        self.useful_ferries=frozenset(f for f,ports in enumerate(self.fs) if len(set(ports)-removed)>=2)
        data=tuple(game['Packages'][p][0] for p in self.package_ids)+tuple(game['Trucks'][t][0] for t in self.truck_ids)+tuple(game['Ferries'][f][0] for f in self.ferry_ids)
        aliases=(self.package_ids,self.truck_ids,self.ferry_ids,tuple(range(self.N)))
        self.initial=self._state(data,aliases)
        self._h_cache={}

    def _tree_setup(self):
        anchors=set(self.gs)|set(self.ferry_served_ports)
        done=set();children=[[] for _ in range(self.N)];roots=[];order=[]
        for start in range(self.N):
            if start in done:continue
            comp={start};stack=[start]
            while stack:
                v=stack.pop()
                for w in self.neighbors[v]:
                    if w not in comp:comp.add(w);stack.append(w)
            done.update(comp)
            if sum(len(self.neighbors[v]) for v in comp)!=2*(len(comp)-1):continue
            root=min(comp&anchors) if comp&anchors else min(comp);roots.append(root)
            stack=[(root,-1)]
            while stack:
                v,parent=stack.pop();order.append(v)
                children[v]=[w for w in self.neighbors[v] if w!=parent]
                stack.extend((w,v) for w in children[v])
        shapes=[-1]*self.N;intern={};enabled=False
        for v in reversed(order):
            key=(v if v in anchors else -1,tuple(sorted(shapes[w] for w in children[v])))
            shapes[v]=intern.setdefault(key,len(intern))
            cs=[shapes[w] for w in children[v]]
            if len(cs)!=len(set(cs)) and (self.N<=24 or any(cs.count(c)>=3 for c in cs)):enabled=True
        self.tree_enabled=enabled
        self.tree_children=children;self.tree_roots=roots;self.tree_order=order;self.tree_shapes=shapes
        self.tree_fixed_children=tuple(tuple(sorted(cs,key=lambda w:(shapes[w],w))) for cs in children)

    def _state(self,data,aliases):
        P,T,F,N=self.P,self.T,self.F,self.N
        q=data[:P];tr=data[P:P+T];fe=data[P+T:]
        cargo=[[] for _ in range(T)];ground=[]
        for i,pos in enumerate(q):
            if pos>=N:cargo[pos-N].append(self.pclass[i])
            else:ground.append((self.pclass[i],pos))
        cargo=[tuple(sorted(x)) for x in cargo]
        lm=None
        if self.tree_enabled:
            pq=[[] for _ in range(N)];tq=[[] for _ in range(N)]
            for c,pos in ground:pq[pos].append(c)
            for t,pos in enumerate(tr):
                if pos<N:tq[pos].append((self.cs[t],cargo[t]))
            dyn=[None]*N;children=self.tree_children;shape=self.tree_shapes
            for v in reversed(self.tree_order):
                dyn[v]=(tuple(sorted(pq[v])),tuple(sorted(tq[v])),tuple(sorted((shape[w],dyn[w]) for w in children[v])))
            lm=list(range(N));stack=[(v,v) for v in self.tree_roots]
            while stack:
                v,target=stack.pop();lm[v]=target
                stack.extend(zip(sorted(children[v],key=lambda w:(shape[w],dyn[w])),self.tree_fixed_children[target]))
        aboard=[[] for _ in range(F)];standing=[]
        for t,pos in enumerate(tr):
            if pos>=N:aboard[pos-N].append((self.cs[t],cargo[t]))
            else:standing.append((self.cs[t],pos if lm is None else lm[pos],cargo[t]))
        if lm is not None:ground=[(c,lm[pos]) for c,pos in ground]
        ships=tuple(sorted((self.fclass[f],port,tuple(sorted(aboard[f]))) for f,port in enumerate(fe)))
        key=(tuple(sorted(ground)),tuple(sorted(standing)),ships)
        return State(data,key,aliases)

    def _h_value(self,state):
        P,T,N=self.P,self.T,self.N;data=state.data
        packages=tuple((p,(self.AT,x) if x<N else (self.IN,self.truck_ids[x-N])) for p,x in zip(self.package_ids,data[:P]))
        trucks=tuple((t,(self.AT,x) if x<N else (self.ON,self.ferry_ids[x-N])) for t,x in zip(self.truck_ids,data[P:P+T]))
        ferries=tuple(zip(self.ferry_ids,data[P+T:]))
        return super()._h_value((packages,trucks,ferries))

    def goal_test(self,state):return state.data[:self.P]==self.gs

    def successor(self,state):
        P,T,N=self.P,self.T,self.N;data=state.data
        q=data[:P];tr=data[P:P+T];fe=data[P+T:]
        pa,ta,fa,la=state.aliases
        def change(i,x):return self._state(data[:i]+(x,)+data[i+1:],state.aliases)
        for p,pos in enumerate(q):
            if pos>=N and tr[pos-N]<N and tr[pos-N]==self.gs[p]:
                t=pos-N
                return [(f'UNLOAD{{{pa[p]},{ta[t]}}}',change(p,tr[t]))]
        active=None
        if self.num_transport_components>1:
            active=min((self.transport_component[self.gs[p]] for p in range(P) if q[p]!=self.gs[p]),default=None)
        load=[0]*T;count=[0]*self.F
        for p,pos in enumerate(q):
            if pos>=N:load[pos-N]+=self.ws[p]
        for pos in tr:
            if pos>=N:count[pos-N]+=1
        standing=set(x for x in tr if x<N)
        useful=set();pickups=set()
        for p,pos in enumerate(q):
            if pos==self.gs[p]:continue
            useful.add(self.gs[p])
            if pos<N:useful.add(pos);pickups.add(pos)
        targets=pickups|self.ferry_served_ports
        out={}
        def add(action,i,x):
            child=change(i,x)
            if child==state:return
            old=out.get(child)
            if old is None or (old[0].startswith('SAIL') and not action.startswith('SAIL')):out[child]=(action,child)
        for f,port in enumerate(fe):
            if f not in self.useful_ferries:continue
            if active is not None and self.transport_component[port]!=active:continue
            for target in self.fs[f]:
                if target not in self.dead_locations and target!=port and (count[f] or target in standing):
                    add(f'SAIL{{{fa[f]},{la[port]},{la[target]}}}',P+T+f,target)
        for p,pos in enumerate(q):
            if active is not None and self.transport_component[self.gs[p]]!=active:continue
            if pos<N:
                if pos==self.gs[p]:continue
                for t in range(T):
                    if tr[t]==pos and load[t]+self.ws[p]<=self.cs[t]:add(f'LOAD{{{pa[p]},{ta[t]}}}',p,N+t)
            else:
                t=pos-N
                if tr[t]<N:add(f'UNLOAD{{{pa[p]},{ta[t]}}}',p,tr[t])
        for t,pos in enumerate(tr):
            loc=pos if pos<N else fe[pos-N]
            if active is not None and self.transport_component[loc]!=active:continue
            if pos<N:
                toward=None
                if load[t]==0:
                    here=self.road_dist[pos];toward=[x for x in targets if here[x]!=float('inf')]
                for target in self.neighbors[pos]:
                    if target in self.dead_locations:continue
                    if toward is not None and not any(self.road_dist[target][x]==here[x]-1 for x in toward):continue
                    far=self.cut_far.get((pos,target))
                    if far is not None and far.isdisjoint(useful):continue
                    add(f'DRIVE{{{ta[t]},{la[pos]},{la[target]}}}',P+t,target)
                for f,port in enumerate(fe):
                    if f in self.useful_ferries and port==pos and count[f]<self.ks[f]:add(f'BOARD{{{ta[t]},{fa[f]}}}',P+t,N+f)
            else:
                f=pos-N;add(f'DISEMBARK{{{ta[t]},{fa[f]}}}',P+t,fe[f])
        return list(out.values())


class LogisticsProblem(_CompactProblem):
    def __init__(self,game):
        super().__init__(game)
        self._ferry_bound_cache={}
        groups={}
        fi={f:i for i,f in enumerate(self.ferry_ids)}
        for f in self.ferry_ids:
            cs=[c for c,fs in self.crossing_ferries.items() if f in fs]
            if cs and fi[f] in self.useful_ferries:groups.setdefault(self.crossing_group[cs[0]],set()).add(fi[f])
        self.vehicle_regions=[]
        for members in groups.values():
            parent=list(range(self.num_islands))
            def find(i):
                while parent[i]!=i:i=parent[i]
                return i
            for f,ports in enumerate(self.fs):
                if f in members:continue
                islands=[self.island[v] for v in ports]
                for x in islands[1:]:parent[find(x)]=find(islands[0])
            roots={};region=[]
            for v in range(self.num_islands):region.append(roots.setdefault(find(v),len(roots)))
            self.vehicle_regions.append((tuple(region),frozenset(members),len(roots)))
    def _h_value(self,state):
        self._bound_data=state.data
        return super()._h_value(state)
    def _sail_lower_bound(self,remaining_islands,ferry_port,package_masks=None):
        P,T,N=self.P,self.T,self.N;data=self._bound_data;q=data[:P];tr=data[P:P+T]
        stock=[0]*self.num_islands
        for x in tr:
            if x<N:stock[self.island[x]]+=1
        flags=tuple(pos>=N and tr[pos-N]>=N for p,pos in enumerate(q) if pos!=self.gs[p])
        key=(tuple(remaining_islands),tuple(self.island[ferry_port[f]] for f in self.ferry_ids),tuple(stock),flags)
        result=self._ferry_bound_cache.get(key)
        if result is None:
            result=self._uncached_sail_bound(remaining_islands,ferry_port,package_masks)
            if len(self._ferry_bound_cache)>200000:self._ferry_bound_cache.clear()
            self._ferry_bound_cache[key]=result
        return result
    def _uncached_sail_bound(self,remaining_islands,ferry_port,package_masks=None):
        sails,rides=super()._sail_lower_bound(remaining_islands,ferry_port,package_masks)
        if not self.F:return sails,rides
        P,T,N=self.P,self.T,self.N;data=self._bound_data;q=data[:P];tr=data[P:P+T];fe=data[P+T:]
        cap=self.max_truck_capacity
        if cap<=0:return sails,rides
        source={};target={};stock=[0]*self.num_islands;aboard=0
        packing=self._packing_loads
        for pos in tr:
            if pos<N:stock[self.island[pos]]+=1
            else:aboard+=1
        for p,pos in enumerate(q):
            if pos==self.gs[p]:continue
            on_ship=False
            if pos<N:loc=pos
            else:
                t=pos-N;loc=tr[t]
                if loc>=N:on_ship=True;loc=fe[loc-N]
            a,b=self.island[loc],self.island[self.gs[p]]
            payload=self._package_bits[self.package_ids[p]] if packing is not None else self.ws[p]
            if a!=b and not on_ship:
                source[a]=(source.get(a,0)|payload) if packing is not None else source.get(a,0)+payload
            if a!=b or on_ship:
                target[b]=(target.get(b,0)|payload) if packing is not None else target.get(b,0)+payload
        def loads(x):return packing[x] if packing is not None else -(-x//cap)
        # Standing trucks obey initial + DISEMBARK = BOARD + final on each
        # island. Each first departure and final arrival carries at most C_max.
        # Any shortage of trucks for departures forces additional disembarks.
        disembarks=sum(max(loads(target.get(v,0)),loads(source.get(v,0))-stock[v]) for v in range(self.num_islands))
        rides=max(rides,disembarks)
        # Cut SETS: remove a ferry group and contract all transport that remains.
        # A package changing contracted regions needs that group, even if no
        # individual crossing is forced. Distinct groups use disjoint ferries.
        group_completed=group_sails=0
        for region,members,nregions in self.vehicle_regions:
            sources={};targets={};supply=[0]*nregions;already=0
            for t,pos in enumerate(tr):
                if pos>=N and pos-N in members:already+=1;continue
                loc=pos if pos<N else fe[pos-N]
                supply[region[self.island[loc]]]+=1
            for p,pos in enumerate(q):
                if pos==self.gs[p]:continue
                on_group=False
                if pos<N:loc=pos
                else:
                    loc=tr[pos-N]
                    if loc>=N:on_group=loc-N in members;loc=fe[loc-N]
                a,b=region[self.island[loc]],region[self.island[self.gs[p]]]
                payload=self._package_bits[self.package_ids[p]] if packing is not None else self.ws[p]
                if a!=b and not on_group:sources[a]=(sources.get(a,0)|payload) if packing is not None else sources.get(a,0)+payload
                if a!=b or on_group:targets[b]=(targets.get(b,0)|payload) if packing is not None else targets.get(b,0)+payload
            completed=sum(max(loads(targets.get(v,0)),loads(sources.get(v,0))-supply[v]) for v in range(nregions))
            group_completed+=completed
            group_sails+=-(-max(0,completed-already)//max(self.ks[f] for f in members))
        rides=max(rides,group_completed)
        sails=max(sails,group_sails)
        # In an optimal plan, a completed ride either already began before this
        # state or has a last SAIL: BOARD...DISEMBARK without sailing is a
        # positive-cost identity and can be removed. A last SAIL serves at most
        # max_ferry_capacity completed rides.
        k=max(self.ks,default=1)
        sails=max(sails,-(-max(0,rides-aboard)//k))
        # A unique ferry serving only two islands must alternate sides. Count
        # its necessary return sails even when the returning sail is empty.
        needed={}
        for index,(a,b,w) in enumerate(remaining_islands):
            if a==b:continue
            for c in self.forced.get((a,b),()):
                if packing is not None and package_masks is not None:needed[c]=needed.get(c,0)|package_masks[index]
                else:needed[c]=needed.get(c,0)+w
        total=0;done=set()
        for a,b in needed:
            if (a,b) in done:continue
            fs=self.crossing_ferries[(a,b)]
            def trips(c):
                payload=needed.get(c,0)
                if packing is not None and package_masks is not None:
                    return -(-packing[payload]//max(self.ferry_capacity[f] for f in self.crossing_ferries[c]))
                capacity=self.crossing_capacity[c]
                return -(-payload//capacity) if capacity else 0
            if len(fs)==1:
                f=next(iter(fs));served={self.island[x] for x in self.ferry_ports[f]}
                if served=={a,b}:
                    na,nb=trips((a,b)),trips((b,a))
                    if self.island[ferry_port[f]]==a:total+=max(0,2*na-1,2*nb)
                    else:total+=max(0,2*na,2*nb-1)
                    done.add((a,b));done.add((b,a));continue
            total+=trips((a,b));done.add((a,b))
        return max(sails,total),rides


def create_logistics_problem(game):
    return LogisticsProblem(game)


if __name__ == '__main__':
    ex1_check.main()
