import numpy as np

np.random.seed(42)

NUM_STOPS = 10

city_coords = np.array([
    [10, 10],
    [ 2,  8],
    [15,  3],
    [ 3,  2],
    [18, 12],
    [11, 19],
    [ 6, 14],
    [19,  7],
    [12,  1],
    [ 1, 17]
])

cost_matrix = np.zeros((NUM_STOPS, NUM_STOPS))
DETOUR_FACTOR = 1.3
AVG_SPEED_KMH = 30.0

for i in range(NUM_STOPS):
    for j in range(NUM_STOPS):
        if i != j:
            euclidean_dist = np.linalg.norm(city_coords[i] - city_coords[j])
            road_distance = euclidean_dist * DETOUR_FACTOR
            travel_time_min = (road_distance / AVG_SPEED_KMH) * 60.0
            cost_matrix[i, j] = travel_time_min
        else:
            cost_matrix[i, j] = np.inf

NUM_ANTS = 10
MAX_ITER = 50
ALPHA = 1.0
BETA = 4.0
RHO = 0.2
Q = 100.0

eta = 1.0 / cost_matrix
np.fill_diagonal(eta, 0)

tau_0 = 1.0 / (NUM_STOPS * np.mean(cost_matrix[cost_matrix != np.inf]))
pheromone = np.full((NUM_STOPS, NUM_STOPS), tau_0)

global_best_tour = None
global_best_cost = np.inf

print("======================================================================")
print("ANT COLONY OPTIMIZATION: URBAN DELIVERY ROUTING")
print("======================================================================")

for iteration in range(1, MAX_ITER + 1):
    ant_tours = []
    ant_costs = []
    
    for ant in range(NUM_ANTS):
        current_node = 0
        visited = [current_node]
        
        while len(visited) < NUM_STOPS:
            unvisited = [node for node in range(NUM_STOPS) if node not in visited]
            
            tau_vals = pheromone[current_node, unvisited] ** ALPHA
            eta_vals = eta[current_node, unvisited] ** BETA
            probabilities = tau_vals * eta_vals
            
            prob_sum = np.sum(probabilities)
            if prob_sum == 0:
                probabilities = np.ones(len(unvisited)) / len(unvisited)
            else:
                probabilities = probabilities / prob_sum
                
            next_node = np.random.choice(unvisited, p=probabilities)
            visited.append(next_node)
            current_node = next_node
            
        visited.append(visited[0])
        
        tour_cost = sum(cost_matrix[visited[i], visited[i+1]] for i in range(NUM_STOPS))
        
        ant_tours.append(visited)
        ant_costs.append(tour_cost)
        
        if tour_cost < global_best_cost:
            global_best_cost = tour_cost
            global_best_tour = visited.copy()
            
    pheromone = (1.0 - RHO) * pheromone
    
    for k in range(NUM_ANTS):
        tour = ant_tours[k]
        cost = ant_costs[k]
        deposit = Q / cost
        for i in range(NUM_STOPS):
            from_node = tour[i]
            to_node = tour[i+1]
            pheromone[from_node, to_node] += deposit
            pheromone[to_node, from_node] += deposit
            
    if iteration % 10 == 0 or iteration == 1:
        print(f"Iter {iteration:2d}/{MAX_ITER} | Best Iteration Time: {min(ant_costs):6.2f} min | Global Best: {global_best_cost:6.2f} min")

stop_names = [
    "Depot", "North Station", "Central Market", "Financial District",
    "East Suburb", "South Hub", "Commercial Zone", "Port Terminal",
    "Industrial Park", "West Mall"
]

print("\n======================================================================")
print("OPTIMAL URBAN DELIVERY ROUTE")
print("======================================================================")
formatted_route = " ──> ".join([f"{stop_names[node]} ({node})" for node in global_best_tour])
print(f"Route Sequence:\n{formatted_route}\n")
print(f"Total Trip Duration: {global_best_cost:.2f} minutes")
print("======================================================================")
