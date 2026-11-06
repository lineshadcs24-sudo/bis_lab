import numpy as np


np.random.seed(42)

NUM_DRONES = 5
DIMENSIONS = NUM_DRONES * 2  
GRID_SIZE = 10000.0          
SENSOR_RADIUS = 1200.0       
NUM_FLORA_POINTS = 500
flora_points = np.random.uniform(0, GRID_SIZE, size=(NUM_FLORA_POINTS, 2))


def evaluate_fitness(position):
    """
    Evaluates coverage and overlap for a given particle's drone layout.
    Lower score is better.
    """
    drones = position.reshape((NUM_DRONES, 2))
    
    diff = flora_points[:, np.newaxis, :] - drones[np.newaxis, :, :]
    distances = np.sqrt(np.sum(diff**2, axis=-1))  # Shape: (500, 5)
    
    covered_points = np.any(distances <= SENSOR_RADIUS, axis=1)
    coverage_ratio = np.sum(covered_points) / NUM_FLORA_POINTS
    
    overlap_penalty = 0.0
    for i in range(NUM_DRONES):
        for j in range(i + 1, NUM_DRONES):
            dist_between_drones = np.linalg.norm(drones[i] - drones[j])
            if dist_between_drones < 2 * SENSOR_RADIUS:
                overlap_penalty += (2 * SENSOR_RADIUS - dist_between_drones) / SENSOR_RADIUS

    unmonitored_penalty = (1.0 - coverage_ratio) * 100.0
    total_cost = unmonitored_penalty + (overlap_penalty * 5.0)
    
    return total_cost, coverage_ratio



SWARM_SIZE = 30
MAX_ITER = 50
w = 0.729   
c1 = 1.494  
c2 = 1.494  
X = np.random.uniform(0, GRID_SIZE, size=(SWARM_SIZE, DIMENSIONS))
V = np.random.uniform(-GRID_SIZE * 0.05, GRID_SIZE * 0.05, size=(SWARM_SIZE, DIMENSIONS))

pbest_positions = X.copy()
pbest_scores = np.zeros(SWARM_SIZE)
pbest_coverages = np.zeros(SWARM_SIZE)

for i in range(SWARM_SIZE):
    cost, cov = evaluate_fitness(X[i])
    pbest_scores[i] = cost
    pbest_coverages[i] = cov

gbest_index = np.argmin(pbest_scores)
gbest_position = pbest_positions[gbest_index].copy()
gbest_score = pbest_scores[gbest_index]
gbest_coverage = pbest_coverages[gbest_index]


print("======================================================================")
print("PSO WILD VEGETATION MONITORING COVERAGE OPTIMIZER")
print("======================================================================")

for iteration in range(1, MAX_ITER + 1):
    for i in range(SWARM_SIZE):
        r1 = np.random.rand(DIMENSIONS)
        r2 = np.random.rand(DIMENSIONS)
        
        cognitive = c1 * r1 * (pbest_positions[i] - X[i])
        social = c2 * r2 * (gbest_position - X[i])
        V[i] = w * V[i] + cognitive + social
        
        X[i] = X[i] + V[i]
        
        X[i] = np.clip(X[i], 0, GRID_SIZE)
        
        current_cost, current_cov = evaluate_fitness(X[i])
        
        if current_cost < pbest_scores[i]:
            pbest_scores[i] = current_cost
            pbest_positions[i] = X[i].copy()
            pbest_coverages[i] = current_cov
            
            if current_cost < gbest_score:
                gbest_score = current_cost
                gbest_position = X[i].copy()
                gbest_coverage = current_cov
                
    if iteration % 10 == 0 or iteration == 1:
        print(f"Iter {iteration:2d}/{MAX_ITER} | Best Cost: {gbest_score:7.2f} | Canopy Coverage: {gbest_coverage*100:5.1f}%")

best_drones = gbest_position.reshape((NUM_DRONES, 2))
print("\n======================================================================")
print("OPTIMAL UAV DEPLOYMENT COORDINATES (X, Y in meters)")
print("======================================================================")
print(f"Total Vegetation Coverage: {gbest_coverage * 100:.2f}%")
for idx, coord in enumerate(best_drones, 1):
    print(f"  - Drone {idx}: [X: {coord[0]:7.1f} m, Y: {coord[1]:7.1f} m]")
print("======================================================================")
