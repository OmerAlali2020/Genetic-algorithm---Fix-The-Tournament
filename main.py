import random
import numpy as np


def create_item(size):
    """
    Create an array of size 32, randomly place numbers from 1 to 32, and divide it into 8 sub-arrays of size 4.

    Returns:
        numpy.ndarray: An array of size 32 divided into 8 sub-arrays of size 4.

    Example:
        >>> create_subarrays()
        array([[12,  5,  7, 10],
               [22, 16,  2, 11],
               [31, 19,  8, 30],
               [13, 25, 17, 23],
               [32, 15,  3,  9],
               [26, 28,  6, 18],
               [20, 21, 27, 24],
               [14,  1,  4, 29]])
    """
    numbers = np.arange(1, size + 1)
    np.random.shuffle(numbers)
    subarrays = numbers.reshape((8, size // 8))
    return subarrays


def create_knockout_decision_matrix(size) -> list:
    matrix = np.random.randint(2, size=(size, size))
    np.fill_diagonal(matrix, -1)

    for i in range(size):
        for j in range(size):
            if i > j:
                if matrix[j][i] == 0:
                    matrix[i][j] = 1
                else:
                    matrix[i][j] = 0

    return matrix.tolist()


def generate_zero_or_one(p):
    choices = [1, 0]
    probabilities = [p, 1 - p]
    result = random.choices(choices, probabilities)[0]
    return result


def create_condorcet_knockout_decision_matrix(size, p):
    matrix = np.zeros((size, size), dtype=int)
    np.fill_diagonal(matrix, -1)

    for i in range(size):
        for j in range(size):

            if i < j:

                # index in the upper third of the matrix

                win = generate_zero_or_one(p)
                matrix[i][j] = win

                if win == 0:
                    matrix[j][i] = 1

    return matrix.tolist()


def print_matrix(matrix):
    """
    Print a matrix in a nicely formatted manner.

    Args:
        matrix (list): The matrix to be printed.

    Example:
        >>> matrix = [[1, 2, 3],
                      [4, 5, 6],
                      [7, 8, 9]]
        >>> print_matrix(matrix)
        1  2  3
        4  5  6
        7  8  9
    """
    for row in matrix:
        print(" ".join(str(element) for element in row))


def groupStage(group, k_d_matrix):
    # run the group stage so each team plays against each team in its stage and the two best teams advance to the
    # next stage

    scores = [0] * len(group)

    for i in range(len(group)):
        for j in range(len(group)):

            if i == j:
                continue

            if k_d_matrix[i][j] == 1:
                scores[i] += 1

    next_stage = []

    max_index = scores.index(max(scores))
    next_stage.append(group[max_index])
    scores[max_index] = -1
    max_index = scores.index(max(scores))
    next_stage.append(group[max_index])

    return next_stage


def is_element_in_array(element, array):
    for sub_array in array:
        if element in sub_array:
            return True
    return False


def fitness_with_prints(item, k_d_matrix, k, knockout_match):
    fitness_score = 1

    # Group stage

    print("\n___Group stage___\n")

    eighth_finals = []

    for i in range(len(item)):
        eighth_finals.append(groupStage(item[i], k_d_matrix))

    print("\nWinners: " + str(eighth_finals))

    flag = is_element_in_array(k, eighth_finals)

    if flag is True:

        fitness_score += 1
    else:
        return fitness_score

    # Knockout stage
    # Eighth final

    print("\n___Eighth final___\n")

    quarter_final = []

    for i in knockout_match:
        # Go through every knockout match in the fixture list, check which team
        # wins and advance them to the quarter final stage.

        team1 = i[0]
        team2 = i[1]

        team_1 = eighth_finals[team1[1] - 1][team1[0]]
        team_2 = eighth_finals[team2[1] - 1][team2[0]]

        print("Game: " + str(team_1) + "-VS-" + str(team_2))

        if k_d_matrix[team_1 - 1][team_2 - 1] == 1:
            print("Winner: " + str(team_1))
            quarter_final.append(team_1)
        else:
            print("Winner: " + str(team_2))
            quarter_final.append(team_2)

    print("\nWinners: " + str(quarter_final))

    if k in quarter_final:
        fitness_score += 1
    else:
        return fitness_score

    # Quarter final stage

    print("\n___Quarter final___\n")

    semifinals = []

    for i in range(0, 8, 2):

        team_1 = quarter_final[i]
        team_2 = quarter_final[i + 1]

        print("Game: " + str(team_1) + "-VS-" + str(team_2))

        if k_d_matrix[team_1 - 1][team_2 - 1] == 1:
            print("Winner: " + str(team_1))
            semifinals.append(team_1)
        else:
            print("Winner: " + str(team_2))
            semifinals.append(team_2)

    print("\nWinners: " + str(semifinals))

    if k in semifinals:
        fitness_score += 1
    else:
        return fitness_score

    # SemiFinal stage

    print("\n___Semifinal___\n")

    final = []

    for i in range(0, 4, 2):

        team_1 = semifinals[i]
        team_2 = semifinals[i + 1]

        print("Game: " + str(team_1) + "-VS-" + str(team_2))

        if k_d_matrix[team_1 - 1][team_2 - 1] == 1:
            print("Winner: " + str(team_1))
            final.append(team_1)
        else:
            print("Winner: " + str(team_2))
            final.append(team_2)

    print("\nWinners: " + str(final))

    if k in final:
        fitness_score += 1
    else:
        return fitness_score

    # Final stage

    print("\n___Final___\n")

    team_1 = final[0]
    team_2 = final[1]

    if k_d_matrix[team_1 - 1][team_2 - 1] == 1:
        print("Final Winner: " + str(team_1))
        winner = team_1
    else:
        print("Final Winner: " + str(team_2))
        winner = team_2

    if k == winner:
        fitness_score += 1

    return fitness_score


def fitness(item, k_d_matrix, k, knockout_match):
    fitness_score = 1

    # Group stage

    eighth_finals = []

    for i in range(len(item)):
        eighth_finals.append(groupStage(item[i], k_d_matrix))

    flag = is_element_in_array(k, eighth_finals)

    if flag is True:

        fitness_score += 1
    else:
        return fitness_score

    # Knockout stage
    # Eighth final

    quarter_final = []

    for i in knockout_match:
        # Go through every knockout match in the fixture list, check which team
        # wins and advance them to the quarter final stage.

        team1 = i[0]
        team2 = i[1]

        team_1 = eighth_finals[team1[1] - 1][team1[0]]
        team_2 = eighth_finals[team2[1] - 1][team2[0]]

        if k_d_matrix[team_1 - 1][team_2 - 1] == 1:

            quarter_final.append(team_1)
        else:

            quarter_final.append(team_2)

    if k in quarter_final:
        fitness_score += 1
    else:
        return fitness_score

    # Quarter final stage

    semifinals = []

    for i in range(0, 8, 2):

        team_1 = quarter_final[i]
        team_2 = quarter_final[i + 1]

        if k_d_matrix[team_1 - 1][team_2 - 1] == 1:

            semifinals.append(team_1)
        else:

            semifinals.append(team_2)

    if k in semifinals:
        fitness_score += 1
    else:
        return fitness_score

    # SemiFinal stage

    final = []

    for i in range(0, 4, 2):

        team_1 = semifinals[i]
        team_2 = semifinals[i + 1]

        if k_d_matrix[team_1 - 1][team_2 - 1] == 1:

            final.append(team_1)
        else:

            final.append(team_2)

    if k in final:
        fitness_score += 1
    else:
        return fitness_score

    # Final stage

    team_1 = final[0]
    team_2 = final[1]

    if k_d_matrix[team_1 - 1][team_2 - 1] == 1:

        winner = team_1
    else:

        winner = team_2

    if k == winner:
        fitness_score += 1

    return fitness_score


def union_subarrays(array):
    union_array = []

    for subarray in array:
        for item in subarray:
            union_array.append(item)

    return union_array


def partially_Matched_Crossover(parent1, parent2):
    parent1 = union_subarrays(parent1)
    parent2 = union_subarrays(parent2)

    n = len(parent1)

    # Choose two cut points at random

    cut1 = np.random.randint(0, n)
    cut2 = np.random.randint(0, n)

    # Ensure cut1 is the smaller cut point

    if cut1 > cut2:
        cut1, cut2 = cut2, cut1

    # Initialize the children

    child1 = [-1] * n
    child2 = [-1] * n

    # Copy the segment between the cut points from parent1 to child1 and from parent2 to child2

    for i in range(cut1, cut2):
        child1[i] = parent1[i]
        child2[i] = parent2[i]

    # Copy the remaining genes from parent2 to child1 and from parent1 to child2

    for i in range(n):
        if i < cut1 or i >= cut2:
            child1[i] = parent2[i]
            child2[i] = parent1[i]

    mapping1 = {}
    mapping2 = {}

    for i in range(cut1, cut2):
        mapping1[parent1[i]] = parent2[i]
        mapping2[parent2[i]] = parent1[i]

    for i in range(n):

        if i < cut1 or i >= cut2:
            while child1[i] in mapping1:
                child1[i] = mapping1[child1[i]]
            while child2[i] in mapping2:
                child2[i] = mapping2[child2[i]]

    child1 = np.reshape(child1, (8, n // 8))
    child2 = np.reshape(child2, (8, n // 8))

    return child1, child2


def scramble_mutation(individual, mutation_rate):
    individual = union_subarrays(individual)
    n = len(individual)

    for i in range(n):

        if random.random() < mutation_rate:

            # Select two random positions in the individual

            pos1 = random.randint(0, len(individual) - 1)
            pos2 = random.randint(0, len(individual) - 1)

            while pos1 == pos2:
                pos2 = random.randint(0, len(individual) - 1)

            # Scramble the values at the selected positions

            individual[pos1], individual[pos2] = individual[pos2], individual[pos1]

    individual = np.reshape(individual, (8, n // 8))

    return individual


def roulette_wheel_selection(population, fitness):
    # Computes the totallity of the population fitness
    population_fitness = sum(fitness)

    # Computes for each chromosome the probability
    chromosome_probabilities = []

    for i in fitness:
        chromosome_probabilities.append(i / population_fitness)

    # Selects one chromosome based on the computed probabilities

    population_index = list(range(0, len(population)))
    choice = np.random.choice(population_index, p=chromosome_probabilities)

    return population[choice]


def create_population(size, item_size):
    """
    Creates a population of individuals.

    Parameters:
        size (int): The desired size of the population.

    Returns:
        list: A list of individuals representing the population.

    Raises:
        ValueError: If size is not a positive integer.

    """
    if not isinstance(size, int) or size <= 0:
        raise ValueError("Size must be a positive integer.")

    population = []

    for _ in range(size):
        population.append(create_item(item_size))

    return population


def evaluate_population_fitness(population, k_d_matrix, k, knockout_match):
    """
          Calculates the fitness of each individual in the population.

          Parameters:
              population (list): A list of individuals representing the population.
              k_d_matrix (list): the result matrix of 1:1 matches between the teams (does team i win/lose to team j)
              k (int): the individual we are interested in in.
              knockout_match (list): the transition function from the house stage to the knockout stage

          Returns:
              List: A list of fitness values corresponding to each individual in the population.

    """

    fitness_array = []

    for item in population:
        fitness_array.append(fitness(item, k_d_matrix, k, knockout_match))

    return fitness_array


def genetic_algorithm(population_size, item_size, k_d_matrix, number_of_generations, mutation_rate,
                      knockout_match_array, selected_individual):
    """
        Executes a genetic algorithm to find the best individual in a population.

        Parameters:
            population_size (int): The size of the population.
            k_d_matrix (list): the result matrix of 1:1 matches between the teams (does team i win/lose to team j)
            number_of_generations (int): The number of generations to run the genetic algorithm.
            mutation_rate (float): The rate at which mutations occur during offspring generation.
            knockout_match_array (list): the transition function from the house stage to the knockout stage
            selected_individual (int): the individual we are interested in.

        Returns:
            tuple: A tuple containing the best individual found and its corresponding score.

        """

    best_score = 0
    best_individual = []

    # Initialize the population

    population = create_population(population_size, item_size)
    population_fitness = evaluate_population_fitness(population, k_d_matrix, selected_individual, knockout_match_array)

    # Do until the selected individual has won the tournament or until the maximum amount of generations

    for i in range(number_of_generations):

        if max(population_fitness) == 6:
            # The individual we want won the tournament

            best_score = max(population_fitness)
            best_individual_index = population_fitness.index(best_score)
            best_individual = population[best_individual_index]

            return best_individual, best_score

        new_population = []

        # Creating a new population

        for j in range(population_size):
            # Selection

            parent1 = roulette_wheel_selection(population, population_fitness)
            parent2 = roulette_wheel_selection(population, population_fitness)

            # Crossover
            offspring = partially_Matched_Crossover(parent1, parent2)[0]

            # Mutation

            offspring = scramble_mutation(offspring, mutation_rate)

            new_population.append(offspring)

        population = new_population
        population_fitness = evaluate_population_fitness(population, k_d_matrix, selected_individual,
                                                         knockout_match_array)

    best_score = max(population_fitness)
    best_individual_index = population_fitness.index(best_score)
    best_individual = population[best_individual_index]

    return best_individual, best_score


knockout_match = [[(0, 1), (0, 2)], [(0, 3), (0, 4)], [(0, 5), (0, 6)], [(0, 7), (0, 8)],
                  [(1, 1), (1, 2)], [(1, 3), (1, 4)], [(1, 5), (1, 6)], [(1, 7), (1, 8)]]

knockout_world_cup = [[(0, 1), (1, 2)], [(0, 3), (1, 4)], [(0, 5), (1, 6)], [(0, 7), (1, 8)],
                      [(0, 2), (1, 1)], [(0, 4), (1, 3)], [(0, 6), (1, 5)], [(0, 8), (1, 7)]]

probabilities = [0.01]
teams = [1]

for t in teams:

    print("T:" + str(t))

    for prob in probabilities:

        scores = []
        times = 100

        for i in range(times):
            m = create_condorcet_knockout_decision_matrix(128, prob)

            a = genetic_algorithm(500, 128, m, 500, 0.1, knockout_match, t)

            scores.append(a[1])

        print(scores.count(6) / times)
