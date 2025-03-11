import math
import os
import random
# Function to read cities from file
def read_cities(file_path="input.txt"):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    num_cities = int(lines[0].strip())  # Number of cities
    cities = []

    for line in lines[1:num_cities+1]:
        data = list(map(int, line.split()))
        x, y, z = data[0], data[1], data[2]
        cities.append((x, y, z))
    
    return cities

# Function to calculate Euclidean distance in 3D space
def euclidean_distance(city1, city2):
    return math.sqrt((city1[0] - city2[0]) ** 2 + (city1[1] - city2[1]) ** 2 + (city1[2] - city2[2]) ** 2)

# Function to calculate total path distance
def path_distance(cities, path):
    total_distance = 0
    for i in range(len(path) - 1):
        total_distance += euclidean_distance(cities[path[i]], cities[path[i+1]])
    return total_distance

#Greedy nearest neighbor algorithm
def nearest_neighbor_path(cities, start_city=None):
    if start_city is None:
        start_city = random.randint(0, len(cities) - 1)  # Random start city
    
    unvisited = set(range(len(cities)))
    path = [start_city]
    unvisited.remove(start_city)
    
    while unvisited:
        last = path[-1]
        next_city = min(unvisited, key=lambda city: euclidean_distance(cities[last], cities[city]))
        path.append(next_city)
        unvisited.remove(next_city)
    
    path.append(path[0])  # Return to the starting city
    return path

def sample(cities):
    population = [nearest_neighbor_path(cities,0)]
    for s in range (1, len(cities) - 1):
        population.append(nearest_neighbor_path(cities,s))
    population = sorted(population, key=lambda path: path_distance(cities, path), reverse=True)
    
## a. Initial Population:
def initialize_population(cities, population_size=0):
    if population_size==0:
        population_size=len(cities)*2
    if population_size>=len(cities)*2:
        population = [nearest_neighbor_path(cities,0)]
        for s in range (1, len(cities) - 1):
            population.append(nearest_neighbor_path(cities,s))
    else:
        population = [nearest_neighbor_path(cities) for _ in range(population_size // 2)]  # Half use nearest neighbor
    for _ in range(population_size // 2):
        path = list(range(len(cities)))
        random.shuffle(path)
        path.append(path[0])  # Form a cycle
        population.append(path)
    return population 

#b. Parent Selection:
#Rank list 
def ComputeRankList(cities, population):
    rank_list = []
    
    for i, path in enumerate(population):
        total_dist = path_distance(cities, path)
        fitness_score = 1 / total_dist 
        rank_list.append((i, fitness_score))
    
    # Sort by fitness score in descending order (higher fitness is better)
    rank_list.sort(key=lambda x: x[1], reverse=True)
    
    return rank_list

#roulette wheel-based selection
def CreateMatingPool(population, RankList):

    # Extract fitness scores
    fitness_scores = [score for _, score in RankList]
    
    # Compute total fitness sum for probability calculation
    total_fitness = sum(fitness_scores)
    
    # Compute selection probabilities
    selection_probs = [score / total_fitness for score in fitness_scores]
    
    # Perform roulette wheel selection
    mating_pool = []
    for _ in range(len(population)):  # Select as many parents as the population size
        pick = random.uniform(0, 1)
        cumulative_prob = 0
        for idx, prob in zip([i for i, _ in RankList], selection_probs):
            cumulative_prob += prob
            if pick <= cumulative_prob:
                mating_pool.append(population[idx])
                break
                
    return mating_pool

def CreateMatingPoolWithElitism(population, RankList, elitism_ratio=0.2):

    # Determine number of elite individuals
    elite_count = max(1, int(len(population) * elitism_ratio))  # At least 1 elite
    
    # Extract elite individuals (directly add the best 10%)
    elite_individuals = [population[idx] for idx, _ in RankList[:elite_count]]
    
    # Extract fitness scores for roulette wheel selection
    fitness_scores = [score for _, score in RankList]
    total_fitness = sum(fitness_scores)
    
    # Compute selection probabilities
    selection_probs = [score / total_fitness for score in fitness_scores]
    
    # Perform roulette wheel selection for the rest
    selected_individuals = []
    for _ in range(len(population) - elite_count):
        pick = random.uniform(0, 1)
        cumulative_prob = 0
        for idx, prob in zip([i for i, _ in RankList], selection_probs):
            cumulative_prob += prob
            if pick <= cumulative_prob:
                selected_individuals.append(population[idx])
                break
    
    # Combine elite individuals with roulette-selected ones
    mating_pool = elite_individuals + selected_individuals
    return mating_pool

#c.crossover
def mutation(path):
    length=(int)(random.uniform(0.2, 0.5)* len(path))
    start=random.randint(0, len(path) - 1-length)
    end=start+length-1;
    new_path= path[:start] + path[start:end+1][::-1] + path[end+1:]
    return new_path;


def Crossover(cities,Parent1, Parent2, Start_index, End_index,m_rate=1,srate=0):
    
    # Step 1: Copy the subarray from Parent1
    child = [None] * len(Parent1)
    child[Start_index:End_index+1] = Parent1[Start_index:End_index+1]
    
    # Step 2: Fill remaining positions with cities from Parent2 while avoiding duplicates
    used_cities = set(child[Start_index:End_index+1])
    child_idx = 0
    
    for city in Parent2:
        if child_idx == Start_index:
            child_idx = End_index + 1  # Skip the copied subarray
        if child_idx >= len(child) - 1:
            break
        
        if city not in used_cities:
            child[child_idx] = city
            used_cities.add(city)
            child_idx += 1
    

    # Ensure the last city is the starting city
    child[-1] = child[0]


        
    if random.random() < m_rate:
        if path_distance(cities,child)>path_distance(cities,mutation(child)):
            child=mutation(child)
    return child

def next_population(cities,population,currentgen,totalgen,elite_rate=0.1):
    new_population = sorted(population, key=lambda path: path_distance(cities, path), reverse=True)[-(len(population) // 10):]
    ranklist=ComputeRankList(cities,population)
    population_size=len(population)
    mating_pool = CreateMatingPoolWithElitism(population, ranklist,elite_rate)  # Select mating pool
    while len(new_population) < population_size:
        parent1, parent2 = random.sample(mating_pool, 2) 
        start, end = sorted(random.sample(range(1, len(parent1) - 1), 2)) 
        child1 = Crossover(cities,parent1, parent2, start, end,1-0.5*currentgen/totalgen,currentgen/totalgen)
        child2 = Crossover(cities,parent2, parent1, start, end,1-0.5*currentgen/totalgen,currentgen/totalgen)

        new_population.extend([child1, child2])
    return new_population[:population_size]
        
# function tests
def PrintMatingPoolDistances(mating_pool, cities):

    # Compute distances and sort in descending order
    sorted_pool = sorted(mating_pool, key=lambda path: path_distance(cities, path), reverse=True)
    
    # print("Mating Pool Distances:")
    # print("-" * 30)
    
    # for i, path in enumerate(sorted_pool, start=1):
    #     distance = path_distance(cities, path)  # Compute total distance of the path
    #     print(f"Individual {i}: Distance {distance:.2f}")
    
    # Print the lowest cost at the end
    lowest_distance = path_distance(cities, sorted_pool[-1])
    # print("-" * 30)
    print(f"Lowest Cost: Distance {lowest_distance:.2f}")


def test():
    # Load cities
    if os.path.exists("input.txt"):
        file_path = "input.txt"
    else:
        file_path = next((file for file in os.listdir() if file.startswith("input")), None)
    if not file_path:
        raise FileNotFoundError("No input file found starting with 'input'")
    cities = read_cities(file_path)

    # Example path (modify as needed)
    #path = list(range(len(cities))); path.append(path[0])  # Example path order based on index
    path =nearest_neighbor_path(cities)
    # Compute total distance
    total_dist = path_distance(cities, path) + euclidean_distance(cities[path[-2]], cities[path[-1]])
    population=initialize_population(cities)
    ranklist=ComputeRankList(cities,population)
    
    # for rank, (idx, fitness) in enumerate(ranklist[:10], start=1):
    #     distance = path_distance(cities, population[idx])  # Get the distance using the index
    #     print(f"Rank {rank}: Index {idx}, Distance {distance:.2f}")
    
    # mating_pool = CreateMatingPoolWithElitism(population, ranklist)  # Select mating pool
    
    PrintMatingPoolDistances(population, cities)  # Print distances
    for i in range(500):
        population=next_population(cities,population)
        print(i)
        PrintMatingPoolDistances(population, cities)
    output_path = "output.txt"
    with open(output_path, "w") as file:
        file.write(f"{total_dist:.2f}\n")
        for i in path:
            file.write(f"{cities[i][0]} {cities[i][1]} {cities[i][2]}\n")
    print(f"Total distance for the given path: {total_dist:.2f} (saved to {output_path})")

    # Example usage:
    Parent1 = [1,2,3,4,5,1]
    Parent2 = [5,2,3,1,4,5]
    Start_index = 2
    End_index = 4

    child = Crossover(cities, Parent1, Parent2, Start_index, End_index)
    print("Child:", child)

    print(mutation(Parent1))
#test()

def genetic_full(generation=500,elite_rate=0.2):
    # Load cities
    if os.path.exists("input.txt"):
        file_path = "input.txt"
    else:
        file_path = next((file for file in os.listdir() if file.startswith("input")), None)
    if not file_path:
        raise FileNotFoundError("No input file found starting with 'input'")
    cities = read_cities(file_path)
    pop_size=2*len(cities)
    generation=500*50//(len(cities))
    pop_size=max(0.2*len(cities),200)
        
    population=initialize_population(cities,pop_size)
    PrintMatingPoolDistances(population, cities)
    
    for i in range(generation):
        population=next_population(cities,population,i,generation,elite_rate)
        print(i)
        PrintMatingPoolDistances(population, cities)
    
    sorted_pool = sorted(population, key=lambda path: path_distance(cities, path), reverse=True)
    total_dist = path_distance(cities, sorted_pool[-1])
    path=sorted_pool[-1]
    output_path = "output.txt"
    with open(output_path, "w") as file:
        file.write(f"{total_dist:.2f}\n")
        for i in path:
            file.write(f"{cities[i][0]} {cities[i][1]} {cities[i][2]}\n")
    print(f"Total distance for the given path: {total_dist:.2f} (saved to {output_path})")

genetic_full()