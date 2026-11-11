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

        # Heaviest load any single truck can carry
        self.max_truck_capacity = max(self.truck_capacity.values(), default=0)

        # Islands and the ferry crossings every route between two islands must use
        self._compute_islands()

        # Roads that cut the map (roads + ferries) in two, used for forced drives
        self._compute_forced_roads()

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

        # ---- Ferries: SAIL ----
        for i, (f, port) in enumerate(ferries):
            for port2 in self.ferry_ports[f]:
                if port2 != port:
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
        for p, pos in packages:
            goal = self.package_goal[p]
            if pos[0] == self.AT:
                if pos[1] != goal:
                    useful.add(pos[1])
                    useful.add(goal)
            else:
                useful.add(goal)

        # ---- Trucks: DRIVE / BOARD / DISEMBARK ----
        cut_far = self.cut_far
        for i, (t, pos) in enumerate(trucks):
            if pos[0] == self.AT:
                l = pos[1]
                # DRIVE along any road from the current location
                for l2 in self.neighbors[l]:
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
        same ferry is an edge of cost crossing_cost. Dijkstra from every location."""
        n = self.num_locations
        edges = {l: {} for l in range(n)}
        for a, b in self.roads:
            edges[a][b] = min(edges[a].get(b, 1), 1)
            edges[b][a] = min(edges[b].get(a, 1), 1)
        for f, served in self.ferry_ports.items():
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
        self._forced_cache = {}

        # (u, v) -> locations on v's side of cut road u-v: the region a truck
        # enters by driving u -> v (it can only enter and leave through u-v)
        self.cut_far = {}
        for (a, b), side in zip(self.cut_roads, self.cut_side):
            self.cut_far[(a, b)] = frozenset(l for l in range(n) if side[l] == 1)
            self.cut_far[(b, a)] = frozenset(l for l in range(n) if side[l] == 0)

    def _forced_drives(self, src, dst):
        """Directed cut roads every route from src to dst must drive:
        a frozenset of (cut road index, direction), direction 0 = first end ->
        second end, 1 = second end -> first end."""
        key = (src, dst)
        res = self._forced_cache.get(key)
        if res is None:
            out = []
            for i, side in enumerate(self.cut_side):
                s, d = side[src], side[dst]
                if s != -1 and d != -1 and s != d:
                    out.append((i, s))           # from side s toward the other
            res = frozenset(out)
            self._forced_cache[key] = res
        return res

    def _mst_bound(self, points, start_cost):
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
                     initial key of every point."""
        if not points:
            return 0
        dist_ns = self.dist_no_sail
        key = {x: start_cost(x) for x in points}
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
        """Lower bound on the number of SAIL actions still needed.
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
        for i, j, w in remaining_islands:
            if i != j:
                workload += w * self.island_hops[i][j]
        cap = self.max_sail_capacity
        by_workload = -(-workload // cap) if cap > 0 and workload else 0

        # Bound 2: forced crossings (below). Both bound the total number of
        # SAILs, so the answer is the larger of the two.
        needed = {}                         # crossing -> weight forced through it
        for i, j, w in remaining_islands:
            if i != j:
                for c in self.forced.get((i, j), ()):
                    needed[c] = needed.get(c, 0) + w
        if not needed:
            return by_workload
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
        return by_forced if by_forced > by_workload else by_workload

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
        and returns a goal distance estimate"""
        packages, trucks, ferries = node.state
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
        # Version A (sails inside the movement terms)
        movement = 0        # max over packages: truck -> package -> goal
        shared_max = 0      # max over non-solo packages: carried trip, full cost
        # Version B (movement terms without any SAIL, sails counted separately)
        movement_ns = 0
        shared_max_ns = 0
        solo_sum = 0        # sum over solo packages: carried trip, no sails (both)
        islands = []        # (current island, goal island) per remaining package
        # directed cut road -> total weight of packages forced through it.
        # A direction with weight 0 is one some truck must drive just to reach
        # a package.
        forced = {}
        must_visit = set()  # locations some truck must still stand at
        truck_locs = {truck_at[t][0] for t in self.truck_ids}
        for p, w, pos in remaining:
            goal = self.package_goal[p]
            if pos[0] == self.AT:
                loc = pos[1]
                load_unload += 2                                # LOAD + UNLOAD
                reach = min(truck_dist(t, loc, dist, 3) for t in self.truck_ids)
                reach_ns = min(truck_dist(t, loc, dist_ns, 0) for t in self.truck_ids)
                carried = dist[loc][goal]
                carried_ns = dist_ns[loc][goal]
                need = reach + carried                          # truck -> package -> goal
                need_ns = reach_ns + carried_ns
                here = island[loc]
                # Cut roads EVERY truck must cross to reach this package: some
                # truck has to drive them toward the package.
                to_pkg = None
                for tl in truck_locs:
                    f = self._forced_drives(tl, loc)
                    to_pkg = f if to_pkg is None else (to_pkg & f)
                    if not to_pkg:
                        break
                if to_pkg:
                    for d in to_pkg:
                        if d not in forced:
                            forced[d] = 0
            else:
                t = pos[1]
                load_unload += 1                                # UNLOAD
                carried = truck_dist(t, goal, dist, 3)
                carried_ns = truck_dist(t, goal, dist_ns, 0)
                need = carried                                  # its truck -> goal
                need_ns = carried_ns
                loc = truck_at[t][0]
                here = island[loc]
            # Cut roads every route of this package to its goal uses: the
            # package's weight must be carried across each of them
            for d in self._forced_drives(loc, goal):
                forced[d] = forced.get(d, 0) + w
            # A truck must stand at the goal to UNLOAD, and at a lying package's
            # location to LOAD it
            must_visit.add(goal)
            if pos[0] == self.AT:
                must_visit.add(loc)
            if need > movement:
                movement = need
            if need_ns > movement_ns:
                movement_ns = need_ns
            islands.append((here, island[goal], w))

            # Solo: cannot share even the biggest truck with any other remaining
            # package. While it is carried its truck holds nothing else, so the
            # DRIVE / BOARD / DISEMBARK actions of its carried trip serve only it.
            other_min = second_w if w == min_w else min_w
            if other_min is not None and w + other_min > cap:
                solo_sum += carried_ns
            else:
                if carried > shared_max:
                    shared_max = carried
                if carried_ns > shared_max_ns:
                    shared_max_ns = carried_ns

        # Version A: carried trips of different solo packages, and of one
        # non-solo package, never share a DRIVE/BOARD/DISEMBARK. Sails are counted
        # only for the non-solo one, so no SAIL is counted twice.
        movement = max(movement, solo_sum + shared_max)
        h_a = load_unload + movement

        # Version B: LOAD/UNLOAD + DRIVE/BOARD/DISEMBARK (no sails) + SAILs.
        # The three parts count different action types, so they can be added.
        # Every directed cut road in `forced` needs its own DRIVE action (one
        # drive can serve many packages, so each is counted once). This and the
        # other movement terms all bound DRIVE/BOARD/DISEMBARK, so take the max.
        # MST over trucks + places to visit: also a bound on DRIVE/BOARD/DISEMBARK
        # (no sails), so it joins the same max.
        truck_ids = self.truck_ids
        mst = self._mst_bound(
            must_visit,
            lambda x: min(truck_dist(t, x, dist_ns, 0) for t in truck_ids))
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
        movement_ns = max(movement_ns, solo_sum + shared_max_ns, forced_drives, mst)
        sails = self._sail_lower_bound(islands, ferry_port)
        h_b = load_unload + movement_ns + 3 * sails

        return h_a if h_a > h_b else h_b


def create_logistics_problem(game):
    print("<<create_logistics_problem")
    """ Create a logistics problem, based on the description.
    game - dictionary as described in pdf file"""
    return LogisticsProblem(game)


if __name__ == '__main__':
    ex1_check.main()
