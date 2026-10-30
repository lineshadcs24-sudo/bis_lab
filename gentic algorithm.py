import random
import copy

# ==========================================
# 1. PROBLEM DATA STRUCTURES & SETUP
# ==========================================

class Exam:
    def __init__(self, exam_id, duration, enrolled_students):
        self.id = exam_id
        self.duration = duration  # in time slots (e.g., 30-min units)
        self.enrolled_students = set(enrolled_students)

class Room:
    def __init__(self, room_id, capacity):
        self.id = room_id
        self.capacity = capacity

class TimetableProblem:
    def __init__(self, exams, rooms, total_slots):
        self.exams = exams         # Dict: exam_id -> Exam object
        self.rooms = rooms         # Dict: room_id -> Room object
        self.total_slots = total_slots

# ==========================================
# 2. CONSTRUCTIVE DECODER & FEASIBILITY
# ==========================================

class TimetableSolution:
    def __init__(self, permutation):
        self.permutation = permutation  # List of exam IDs in priority order
        self.schedule = {}              # exam_id -> {'slot': int, 'room': room_id}
        self.fitness = float('inf')
        self.hard_violations = 0
        self.soft_penalties = 0

def decode_permutation(permutation, problem):
    """
    Constructively maps an exam permutation into a valid timetable schedule.
    Enforces hard constraints (student clashes, room capacity, slot bounds).
    """
    solution = TimetableSolution(permutation)
    student_occupancy = {}  # student_id -> set of occupied slots
    room_occupancy = {}     # (room_id, slot) -> bool
    
    unplaced_count = 0

    for exam_id in permutation:
        exam = problem.exams[exam_id]
        placed = False
        
        # Try to find the earliest valid time slot and room
        for slot in range(problem.total_slots):
            # Check if exam exceeds time horizon
            if slot + exam.duration > problem.total_slots:
                continue
            
            exam_slots = set(range(slot, slot + exam.duration))
            
            # Check student clashes
            has_student_clash = False
            for student in exam.enrolled_students:
                if student in student_occupancy:
                    if not student_occupancy[student].isdisjoint(exam_slots):
                        has_student_clash = True
                        break
            if has_student_clash:
                continue
                
            # Find an available room with sufficient capacity
            suitable_room = None
            for room_id, room in problem.rooms.items():
                if room.capacity < len(exam.enrolled_students):
                    continue  # Room too small (hard constraint)
                
                # Check room availability across all required slots
                room_free = True
                for s in exam_slots:
                    if room_occupancy.get((room_id, s), False):
                        room_free = False
                        break
                if room_free:
                    suitable_room = room_id
                    break
            
            if suitable_room is not None:
                # Assign exam
                solution.schedule[exam_id] = {'slot': slot, 'room': suitable_room}
                # Lock student slots
                for student in exam.enrolled_students:
                    if student not in student_occupancy:
                        student_occupancy[student] = set()
                    student_occupancy[student].update(exam_slots)
                # Lock room slots
                for s in exam_slots:
                    room_occupancy[(suitable_room, s)] = True
                
                placed = True
                break
        
        if not placed:
            unplaced_count += 1
            # Default fallback assignment if it cannot be legally placed
            solution.schedule[exam_id] = {'slot': 0, 'room': list(problem.rooms.keys())[0]}

    solution.hard_violations = unplaced_count
    return solution

# ==========================================
# 3. FITNESS FUNCTION
# ==========================================

def evaluate_fitness(solution, problem):
    """
    Calculates the penalty score. Hard violations are heavily weighted.
    Soft constraints penalize back-to-back exams or high daily concentration.
    """
    weights = {
        'hard_violation': 10000,
        'back_to_back': 100,
        'daily_concentration': 50
    }
    
    score = solution.hard_violations * weights['hard_violation']
    
    # Track student schedules for soft constraints (e.g., back-to-back exams)
    student_slots = {}
    for exam_id, details in solution.schedule.items():
        exam = problem.exams[exam_id]
        start = details['slot']
        for student in exam.enrolled_students:
            if student not in student_slots:
                student_slots[student] = []
            student_slots[student].append((start, start + exam.duration))
            
    # Soft constraint check: Back-to-back exams
    back_to_back_count = 0
    for student, exams_times in student_slots.items():
        exams_times.sort(key=lambda x: x[0])
        for i in range(len(exams_times) - 1):
            if exams_times[i][1] == exams_times[i+1][0]: # Ends exactly when next starts
                back_to_back_count += 1
                
    score += back_to_back_count * weights['back_to_back']
    solution.soft_penalties = back_to_back_count * weights['back_to_back']
    solution.fitness = score
    return score

# ==========================================
# 4. GENETIC OPERATORS (SELECTION, CROSSOVER, MUTATION)
# ==========================================

def tournament_selection(population, k=3):
    """Selects the fittest individual from a random tournament group of size k."""
    subset = random.sample(population, k)
    return min(subset, key=lambda ind: ind.fitness)

def order_crossover(parent1, parent2):
    """Performs Order Crossover (OX) to preserve permutation validity."""
    size = len(parent1)
    child_perm = [None] * size
    
    # Pick a random subsegment from parent 1
    start, end = sorted(random.sample(range(size), 2))
    child_perm[start:end+1] = parent1[start:end+1]
    
    # Fill remaining slots from parent 2 in order
    p2_idx = 0
    for i in range(size):
        if child_perm[i] is None:
            while parent2[p2_idx] in child_perm:
                p2_idx += 1
            child_perm[i] = parent2[p2_idx]
            
    return child_perm

def mutate(permutation, swap_prob=0.05):
    """Applies pairwise swap mutation."""
    mutated = list(permutation)
    if random.random() < swap_prob:
        idx1, idx2 = random.sample(range(len(mutated)), 2)
        mutated[idx1], mutated[idx2] = mutated[idx2], mutated[idx1]
    return mutated

# ==========================================
# 5. MAIN GENETIC ALGORITHM ENGINE
# ==========================================

def run_genetic_algorithm(problem, pop_size=30, generations=50):
    exam_ids = list(problem.exams.keys())
    population = []
    for _ in range(pop_size):
        perm = list(exam_ids)
        random.shuffle(perm)
        ind = decode_permutation(perm, problem)
        evaluate_fitness(ind, problem)
        population.append(ind)
        
    best_overall = min(population, key=lambda x: x.fitness)
    
    for gen in range(generations):
        new_population = []
        
        # Elitism: carry over the best individual
        best_overall = min(population, key=lambda x: x.fitness)
        new_population.append(copy.deepcopy(best_overall))
        
        while len(new_population) < pop_size:
            p1 = tournament_selection(population)
            p2 = tournament_selection(population)
            
            child_perm = order_crossover(p1.permutation, p2.permutation)
            child_perm = mutate(child_perm)
            
            child = decode_permutation(child_perm, problem)
            evaluate_fitness(child, problem)
            new_population.append(child)
            
        population = new_population
        current_best = min(population, key=lambda x: x.fitness)
        if current_best.fitness < best_overall.fitness:
            best_overall = copy.deepcopy(current_best)
            
        print(f"Generation {gen+1}/{generations} | Best Fitness Penalty: {best_overall.fitness}")
        
    return best_overall

# ==========================================
# EXAMPLE EXECUTION SCRIPT
# ==========================================
if __name__ == "__main__":
    exams_data = {
        'CS101': Exam('CS101', duration=2, enrolled_students=[1, 2, 3, 4, 5]),
        'MATH201': Exam('MATH201', duration=2, enrolled_students=[3, 4, 6, 7]),
        'PHYS102': Exam('PHYS102', duration=3, enrolled_students=[1, 5, 6, 8]),
        'ENG103': Exam('ENG103', duration=2, enrolled_students=[2, 7, 8]),
        'HIST101': Exam('HIST101', duration=2, enrolled_students=[1, 3, 8])
    }
    
    rooms_data = {
        'Room_A': Room('Room_A', capacity=10),
        'Room_B': Room('Room_B', capacity=10)
    }
    
    problem_instance = TimetableProblem(exams=exams_data, rooms=rooms_data, total_slots=16)
    
    print("Running Exam Timetabling Genetic Algorithm...")
    best_schedule = run_genetic_algorithm(problem_instance, pop_size=10, generations=15)
    
    print("\n--- Optimization Complete ---")
    print(f"Final Optimal Penalty Score: {best_schedule.fitness}")
    print("Assigned Schedule:")
    for exam_id, details in best_schedule.schedule.items():
        print(f"  Exam {exam_id} -> Time Slot: {details['slot']}, Room: {details['room']}")