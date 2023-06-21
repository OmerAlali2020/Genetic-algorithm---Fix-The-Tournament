import random
import numpy as np


def create_item(size, number_of_groups):
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
    sub_arrays = numbers.reshape((number_of_groups, size // number_of_groups))
    return sub_arrays


def create_item_by_tiers():
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

    groups = 8
    tiers = 4
    item = []

    tier_1 = np.arange(1, 9)
    tier_2 = np.arange(9, 17)
    tier_3 = np.arange(17, 25)
    tier_4 = np.arange(25, 33)

    for i in range(groups):
        group = []

        team_1 = random.choice(tier_1)
        tier_1 = np.delete(tier_1, np.argwhere(tier_1 == team_1))
        group.append(team_1)
        team_2 = random.choice(tier_2)
        tier_2 = np.delete(tier_2, np.argwhere(tier_2 == team_2))
        group.append(team_2)
        team_3 = random.choice(tier_3)
        tier_3 = np.delete(tier_3, np.argwhere(tier_3 == team_3))
        group.append(team_3)
        team_4 = random.choice(tier_4)
        tier_4 = np.delete(tier_4, np.argwhere(tier_4 == team_4))
        group.append(team_4)

        item.append(group)

    return item


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


def create_fifa_knockout_decision_matrix(p_matrix):

    size = len(p_matrix)
    matrix = np.zeros((size, size), dtype=int)
    np.fill_diagonal(matrix, -1)

    for i in range(size):
        for j in range(size):

            if i < j:

                # index in the upper third of the matrix

                win = generate_zero_or_one(p_matrix[i][j])
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

            if k_d_matrix[group[i]-1][group[j]-1] == 1:
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


def fitness_with_prints(item, k_d_matrix, k, knockout_match, fifa_teams_names):
    fitness_score = 1

    # Group stage

    print("\n___Group stage___\n")

    eighth_finals = []

    for i in range(len(item)):
        eighth_finals.append(groupStage(item[i], k_d_matrix))

    print("\nWinners: ")

    for final_team in eighth_finals:
        print(fifa_teams_names[final_team[0]], fifa_teams_names[final_team[1]])

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

        print("Game: " + fifa_teams_names[team_1] + "-VS-" + fifa_teams_names[team_2])

        if k_d_matrix[team_1 - 1][team_2 - 1] == 1:
            print("Winner: " + fifa_teams_names[team_1])
            quarter_final.append(team_1)
        else:
            print("Winner: " + fifa_teams_names[team_2])
            quarter_final.append(team_2)

    print("\nWinners: ")
    for final_team in quarter_final:
        print(fifa_teams_names[final_team])

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

        print("Game: " + fifa_teams_names[team_1] + "-VS-" + fifa_teams_names[team_2])

        if k_d_matrix[team_1 - 1][team_2 - 1] == 1:
            print("Winner: " + fifa_teams_names[team_1])
            semifinals.append(team_1)
        else:
            print("Winner: " + fifa_teams_names[team_2])
            semifinals.append(team_2)

    print("\nWinners: ")
    for final_team in semifinals:
        print(fifa_teams_names[final_team])

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

        print("Game: " + fifa_teams_names[team_1] + "-VS-" + fifa_teams_names[team_2])

        if k_d_matrix[team_1 - 1][team_2 - 1] == 1:
            print("Winner: " + fifa_teams_names[team_1])
            final.append(team_1)
        else:
            print("Winner: " + fifa_teams_names[team_2])
            final.append(team_2)

    print("\nWinners: " + str(final))
    for final_team in final:
        print(fifa_teams_names[final_team])

    if k in final:
        fitness_score += 1
    else:
        return fitness_score

    # Final stage

    print("\n___Final___\n")

    team_1 = final[0]
    team_2 = final[1]

    if k_d_matrix[team_1 - 1][team_2 - 1] == 1:
        print("Final Winner: " + fifa_teams_names[team_1])
        winner = team_1
    else:
        print("Final Winner: " + fifa_teams_names[team_2])
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


def fitness_new_format(item, k_d_matrix, k, knockout_match):

    # TODO check new fittness function step by step with prints

    fitness_score = 1

    # Group stage

    round_of_32 = []

    for i in range(len(item)):
        round_of_32.append(groupStage(item[i], k_d_matrix))

    flag = is_element_in_array(k, round_of_32)

    if flag is True:

        fitness_score += 1
    else:
        return fitness_score

    # Knockout stage - Round of 16

    eighth_final = []

    for i in knockout_match:
        # Go through every knockout match in the fixture list, check which team
        # wins and advance them to the quarter final stage.

        team1 = i[0]
        team2 = i[1]

        team_1 = round_of_32[team1[1] - 1][team1[0]]
        team_2 = round_of_32[team2[1] - 1][team2[0]]

        if k_d_matrix[team_1 - 1][team_2 - 1] == 1:

            eighth_final.append(team_1)
        else:

            eighth_final.append(team_2)

    if k in eighth_final:
        fitness_score += 1
    else:
        return fitness_score

    # Eighth - finals stage

    quarter_final = []

    for i in range(0, 16, 2):

        team_1 = eighth_final[i]
        team_2 = eighth_final[i + 1]

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

    for i in range (0, 8, 2):

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

    number_of_groups = len(parent1)

    # Union subarray to enable PMX operate on the item
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

    child1 = np.reshape(child1, (number_of_groups, n // number_of_groups))
    child2 = np.reshape(child2, (number_of_groups, n // number_of_groups))

    return child1, child2


def scramble_mutation(individual, mutation_rate):

    if random.random() < mutation_rate:

        number_of_groups = len(individual)
        number_of_teams = len(individual[0])

        for i in range(number_of_groups):

            position = random.randint(0, number_of_teams-1)
            destination_group = random.randint(0, number_of_groups-1)

            while i == destination_group:
                destination_group = random.randint(0, number_of_groups-1)

            # Scramble the values at the selected positions
            placeholder = individual[i][position]
            individual[i][position] = individual[destination_group][position]
            individual[destination_group][position] = placeholder

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


def create_population(size, item_size, number_of_groups):

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

        # TODO Change back the function to create_item or change create_population to get many types of
        #  create item function
        population.append(create_item(item_size, number_of_groups))

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


def genetic_algorithm(population_size, item_size, number_of_groups, k_d_matrix, number_of_generations, mutation_rate,
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

    population = create_population(population_size, item_size, number_of_groups)
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


def calculate_probability_by_fifa_scores(fifa_scores, team_1, team_2):

    # TODO Consider optimizing the function by taking the calculation of the minimum difference out

    n = len(fifa_scores)
    sorted_scores = sorted(fifa_scores)

    max_d = sorted_scores[n - 1] - sorted_scores[0]
    min_d = sorted_scores[1] - sorted_scores[0]

    # Calculate the minimum difference between any 2 teams scores
    for i in range(2, n):
        min_d = min(min_d, sorted_scores[i] - sorted_scores[i - 1])

    # Calculate the probabilities by normalized the differences to 0.5 <= d <= 1.0

    d = abs(fifa_scores[team_1] - fifa_scores[team_2])
    d_norm = ((d - min_d) / (max_d - min_d)) * (1 - 0.5) + 0.5

    if fifa_scores[team_1] > fifa_scores[team_2]:

        return d_norm

    else:

        return 1 - d_norm


def create_fifa_probability_matrix(fifa_scores):

    # TODO add documentation that explain why we fill only the upper third of the matrix
    size = len(fifa_scores)
    matrix = np.full((size, size), 0.0)
    np.fill_diagonal(matrix, -1)

    for i in range(size):
        for j in range(size):

            if i < j:

                matrix[i][j] = calculate_probability_by_fifa_scores(fifa_scores, i, j)

    return matrix.tolist()


knockout_world_cup = [[(0, 1), (1, 2)], [(0, 3), (1, 4)], [(0, 5), (1, 6)], [(0, 7), (1, 8)],
                      [(0, 2), (1, 1)], [(0, 4), (1, 3)], [(0, 6), (1, 5)], [(0, 8), (1, 7)]]

knockout_new_format_world_cup = [[(0, 1), (1, 2)], [(0, 3), (1, 4)], [(0, 5), (1, 6)], [(0, 7), (1, 8)],
                                 [(0, 2), (1, 1)], [(0, 4), (1, 3)], [(0, 6), (1, 5)], [(0, 8), (1, 7)],
                                [(0, 9), (1, 10)], [(0, 11), (1, 12)], [(0, 13), (1, 14)], [(0, 15), (1, 16)],
                                 [(0, 10), (1, 9)], [(0, 12), (1, 11)], [(0, 14), (1, 13)], [(0, 16), (1, 15)]]


fifa_scores = [1388.61, 1834.21, 1792.53, 1838.45, 1840.93, 1792.43, 1682.85, 1707.22,
               1631.87, 1731.23, 1594.53, 1647.42, 1631.29, 1664.24, 1653.77, 1730.02,
               1613.21, 1553.23, 1588.59, 1677.79, 1541.52, 1553.76, 1536.01, 1535.76,
               1470.21, 1442.66, 1478.13, 1421.46, 1396.01, 1538.95, 1491.12, 1532.79]

# TODO mabey delete None and Chane fitness with print to final_team - 1
fifa_teams = ['None', 'Qatar', 'Brazil', 'Belgium', 'France', 'Argentina', 'England', 'Spain', 'Portugal',
              'Mexico','Netherlands', 'Denmark', 'Germany', 'Uruguay', 'Switzerland', 'United States', 'Croatia',
              'Senegal', 'Iran', 'Japan', 'Morocco', 'Serbia', 'Poland', 'South Korea', 'Tunisia',
              'Cameroon', 'Canada', 'Ecuador', 'Saudi Arabia', 'Ghana', 'Wales', 'Costa Rica', 'Australia'
              ]

times = 100
# [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
# 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 1]

teams = [32]
prob = [0.4, 0.2, 0.1, 0.05, 0.01]
item_size = 32
num_of_groups = 8

"""
for t in teams:

    print("T: " + str(t))

    for p in prob:

        scores = []

        for i in range(times):

            m = create_condorcet_knockout_decision_matrix(128, p)
            a = genetic_algorithm(100, item_size ,num_of_groups, m, 10, 0.05, knockout_world_cup, t)
        
            scores.append(a[1])

        print(scores.count(6) / times)

"""




