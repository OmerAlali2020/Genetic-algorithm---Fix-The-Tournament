import random
import numpy as np


def create_item(size, number_of_groups):
    """
    Create an array of size 'size' divided into 'number_of_groups' subarrays of size 'size // number_of_groups'.

    Args:
        size (int): The number of teams in the tournament.
        number_of_groups (int): The number of groups to divide the teams into.

    Returns:
        np.ndarray: An array of size 'size' divided into 'number_of_groups' subarrays of size 'size // number_of_groups'

    Example:
        >>> create_item(32, 8)
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
    Generate team arrangements based on tiers.

    Returns:
    list: A list of team arrangements in size 32. Each arrangement is a list of teams, where in each group
    there are 4 groups from 4 different tiers

    Example:
        >>> create_item_by_tiers()
        [[6, 10, 23, 30],
         [7, 14, 21, 28],
         [5, 13, 19, 29],
         [1, 12, 18, 26],
         [4, 9, 22, 25],
         [2, 16, 17, 32],
         [8, 11, 15, 20],
         [3, 24, 27, 31]]
    """

    groups = 8
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


def create_knockout_decision_matrix(size):
    """
    Create a matrix at size 'size' and determine randomly at each index i,j whether group i beats j.

    Returns:
    list: a matrix as a list of size 'size'. A win is 1,
    a loss is 0. The diagonal is coded to -1.


    Example:
        >>> create_knockout_decision_matrix(2)
        [[-1, 0],
        [1, -1]]
    """

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
    """
    generate 0 or 1 by probabilty p

    Args:
    p (float): The probabilty to get 1

    Returns:
    int: 0 or 1

    Example:
    >>> generate_zero_or_one(0.3)
    0
    """

    choices = [1, 0]
    probabilities = [p, 1 - p]
    result = random.choices(choices, probabilities)[0]
    return result


def create_condorcet_knockout_decision_matrix(size, p):
    """

    Create a matrix of size 'size' and determine with probability p for each index i,j if team i beats j,
    under Condresa model

    Args:
    size (int): The matrix size
    p (float): The probabilty to get 1

    Returns:
    list: a matrix as a list of size 'size'. A win is 1, a loss is 0. The diagonal is
    coded to -1.


    Example:
        >>> create_condorcet_knockout_decision_matrix(2, 0.2)
        [[-1, 0],
        [1, -1]]
    """

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
    """
    Create a matrix of size 'size' and determine with probability p by p_matrix for each index i,j if team i beats j,
    under Condresa model

    Args:
    p_matrix (list): Probability matrix of victories for all two teams i,j according to FIFA data

    Returns:
    list: a matrix as a list of size 'size'. A win is 1, a loss is 0. The diagonal is
    coded to -1.


    Example:
        >>> create_fifa_knockout_decision_matrix(m)
        [[-1, 0],
        [1, -1]]
    """

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
        >>> print_matrix(matrix)
        1  2  3
        4  5  6
        7  8  9
    """
    for row in matrix:
        print(" ".join(str(element) for element in row))


def groupStage(group, k_d_matrix):
    """
    Run the group stage so each team plays against each team in its stage and return the two highest scoring teams

    Args:
    group (list): The list of the teams that are in the group
    k_d_matrix (list): A victory matrix where in each index it is determined whether team i wins against team j

    Returns:
    list: The two teams that advanced to the next stage.

    Example:
    >>> groupStage(group, matrix)
    (1,17)
    """

    # run the group stage so each team plays against each team in its stage

    scores = [0] * len(group)

    for i in range(len(group)):
        for j in range(len(group)):

            if i == j:
                continue

            if k_d_matrix[group[i] - 1][group[j] - 1] == 1:
                scores[i] += 1

    # Choose the two teams that got the highest score

    next_stage = []

    max_index = scores.index(max(scores))
    next_stage.append(group[max_index])
    scores[max_index] = -1
    max_index = scores.index(max(scores))
    next_stage.append(group[max_index])

    return next_stage


def is_element_in_array(element, array):
    """
    Check if an element is present in a nested array.

    Args:
        element: The element to search for.
        array: The nested array to search in.

    Returns:
        bool: True if the element is found in the array, False otherwise.

    Example:
        >>> is_element_in_array(4, [[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        True

        >>> is_element_in_array(10, [[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        False
    """
    for sub_array in array:
        if element in sub_array:
            return True
    return False


def knockout_round(previous_round_winners, k_d_matrix):

    next_round_winners = []

    for i in range(0, len(previous_round_winners), 2):

        team_1 = previous_round_winners[i]
        team_2 = previous_round_winners[i + 1]

        if k_d_matrix[team_1 - 1][team_2 - 1] == 1:

            next_round_winners.append(team_1)
        else:

            next_round_winners.append(team_2)

    return next_round_winners


def fitness(item, k_d_matrix, k, knockout_match, number_of_knockout_rounds):

    fitness_score = 1

    # Run the Group stage

    group_stage_winners = []

    for i in range(len(item)):
        group_stage_winners.append(groupStage(item[i], k_d_matrix))

    flag = is_element_in_array(k, group_stage_winners)

    if flag is True:

        fitness_score += 1
    else:
        return fitness_score

    # Knockout stage

    first_knockout_round_winners = []

    # In the first knockout round, teams play against each other based on the knockout game array,
    # which determines which team plays against which

    for i in knockout_match:

        team1 = i[0]
        team2 = i[1]

        team_1 = group_stage_winners[team1[1] - 1][team1[0]]
        team_2 = group_stage_winners[team2[1] - 1][team2[0]]

        if k_d_matrix[team_1 - 1][team_2 - 1] == 1:

            first_knockout_round_winners.append(team_1)
        else:

            first_knockout_round_winners.append(team_2)

    if k in first_knockout_round_winners:
        fitness_score += 1
    else:
        return fitness_score

    # Play additional knockout rounds according to the requested format (3, 4, etc.)

    for i in range(number_of_knockout_rounds - 1):

        new_knockout_round_winners = knockout_round(first_knockout_round_winners, k_d_matrix)

        if k in new_knockout_round_winners:
            fitness_score += 1
            first_knockout_round_winners = new_knockout_round_winners
        else:
            return fitness_score

    return fitness_score


def union_subarrays(array):
    """
    Flatten a nested array and return the union of all subarrays.

    Args:
        array: The nested array to flatten.

    Returns:
        list: A list containing all the elements from the subarrays.

    Example:
        >>> union_subarrays([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        [1, 2, 3, 4, 5, 6, 7, 8, 9]
    """
    union_array = []

    for subarray in array:
        for item in subarray:
            union_array.append(item)

    return union_array


def partially_Matched_Crossover(parent1, parent2):
    """
    Perform Partially Matched Crossover (PMX) on two parent arrays.

    Args:
        parent1: The first parent array.
        parent2: The second parent array.

    Returns:
        tuple: A tuple containing two child arrays resulting from PMX.

    Example:
        >>> parent1 = [[1,2,3],[5,4,6]]
        >>> parent2 = [[6,5,4], [3,2,1]]
        >>> partially_Matched_Crossover(parent1, parent2)
        ([[6, 2, 3], [5, 4, 1]], [[1, 5, 4], [3, 2, 6]])
    """
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
    """
    Apply scramble mutation to an individual array.

    Args:
        individual: The individual array to mutate.
        mutation_rate: The probability of mutation for each element.

    Returns:
        list: The mutated individual array.

    Example:
        >>> individual = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        >>> mutation_rate = 0.1
        >>> scramble_mutation(individual, mutation_rate)
        [[1, 2, 3], [4, 5, 6], [7, 9, 8]]
    """
    if random.random() < mutation_rate:
        number_of_groups = len(individual)
        number_of_teams = len(individual[0])

        for i in range(number_of_groups):
            position = random.randint(0, number_of_teams - 1)
            destination_group = random.randint(0, number_of_groups - 1)

            while i == destination_group:
                destination_group = random.randint(0, number_of_groups - 1)

            # Scramble the values at the selected positions
            placeholder = individual[i][position]
            individual[i][position] = individual[destination_group][position]
            individual[destination_group][position] = placeholder

    return individual


def roulette_wheel_selection(population, fitness):
    """
    Perform roulette wheel selection to choose an individual from the population based on fitness.

    Args:
        population: The population of individuals.
        fitness: The fitness values corresponding to each individual in the population.

    Returns:
        list: The selected individual from the population.

    Example:
        >>> population = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        >>> fitness = [0.1, 0.4, 0.5]
        >>> roulette_wheel_selection(population, fitness)
        [7, 8, 9]
    """
    # Compute the total fitness of the population
    population_fitness = sum(fitness)

    # Compute the probability for each chromosome
    chromosome_probabilities = [i / population_fitness for i in fitness]

    # Select one chromosome based on the computed probabilities
    population_index = list(range(len(population)))
    choice = np.random.choice(population_index, p=chromosome_probabilities)

    return population[choice]


def create_population(size, item_size, number_of_groups):
    """
    Creates a population of individuals.

    Parameters:
        size (int): The desired size of the population.
        item_size (int): The number of teams of each item
        number_of_groups: The number of groups of each item

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


def evaluate_population_fitness(population, k_d_matrix, k, knockout_match, number_of_knockout_rounds):
    """
          Calculates the fitness of each individual in the population.

          Parameters:
              population (list): A list of individuals representing the population.
              k_d_matrix (list): the result matrix of 1:1 matches between the teams (does team i win/lose to team j)
              k (int): the individual we are interested in in.
              knockout_match (list): the transition function from the house stage to the knockout stage
              number_of_knockout_rounds (int): The number of knockout rounds in the game format

          Returns:
              List: A list of fitness values corresponding to each individual in the population.

    """

    fitness_array = []

    for item in population:
        fitness_array.append(fitness(item, k_d_matrix, k, knockout_match, number_of_knockout_rounds))

    return fitness_array


def genetic_algorithm(population_size, item_size, number_of_groups, k_d_matrix, number_of_generations, mutation_rate,
                      knockout_match_array, selected_individual, knockout_rounds):
    """
        Executes a genetic algorithm to find the best individual in a population.

        Parameters:
            population_size (int): The size of the population.
            item_size (int): the number of teams of each item
            number_of_groups (int): the number of groups of each item
            k_d_matrix (list): the result matrix of 1:1 matches between the teams (does team i win/lose to team j)
            number_of_generations (int): The number of generations to run the genetic algorithm.
            mutation_rate (float): The rate at which mutations occur during offspring generation.
            knockout_match_array (list): the transition function from the house stage to the knockout stage
            selected_individual (int): the individual we are interested in.
            knockout_rounds (int): The number of knockout rounds in the game

        Returns:
            tuple: A tuple containing the best individual found [0] and its corresponding score [1]

        """

    best_score = 0
    best_individual = []

    # Initialize the population

    population = create_population(population_size, item_size, number_of_groups)
    population_fitness = evaluate_population_fitness(population, k_d_matrix, selected_individual, knockout_match_array,
                                                     knockout_rounds)

    # Do until the selected individual has won the tournament or until the maximum amount of generations

    max_score = knockout_rounds + 2

    for i in range(number_of_generations):

        if max(population_fitness) == max_score:
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
                                                         knockout_match_array, knockout_rounds)

    best_score = max(population_fitness)
    best_individual_index = population_fitness.index(best_score)
    best_individual = population[best_individual_index]

    return best_individual, best_score


def calculate_probability_by_fifa_scores(fifa_scores, team_1, team_2):
    """
    Calculate the probability of a team winning based on FIFA scores.

    Args:
        fifa_scores (list): The list of FIFA scores for each team.
        team_1 (int): The ID of the first team.
        team_2 (int): The ID of the second team.

    Returns:
        float: The probability of the first team winning against the second team.

    Example:
        >>> fifa_scores = [90, 85, 88, 92, 87]
        >>> team_1 = 1
        >>> team_2 = 3
        >>> calculate_probability_by_fifa_scores(fifa_scores, team_1, team_2)
        0.75
    """
    n = len(fifa_scores)
    sorted_scores = sorted(fifa_scores)

    max_d = sorted_scores[n - 1] - sorted_scores[0]
    min_d = sorted_scores[1] - sorted_scores[0]

    # Calculate the minimum difference between any 2 teams' scores
    for i in range(2, n):
        min_d = min(min_d, sorted_scores[i] - sorted_scores[i - 1])

    # Calculate the probabilities by normalizing the differences to 0.5 <= d <= 1.0

    d = abs(fifa_scores[team_1] - fifa_scores[team_2])
    d_norm = ((d - min_d) / (max_d - min_d)) * 0.5 + 0.5

    if fifa_scores[team_1] > fifa_scores[team_2]:
        return d_norm
    else:
        return 1 - d_norm


def create_fifa_probability_matrix(fifa_scores):
    """
    Create a probability matrix based on FIFA scores.

    Args:
        fifa_scores (list): The list of FIFA scores for each team.

    Returns:
        list: The probability matrix where each element represents the probability of the corresponding teams' match.

    Example:
        >>> fifa_scores = [90, 85, 88, 92]
        >>> create_fifa_probability_matrix(fifa_scores)
        [[-1, 0.6, 0.75, 0.4],
         [-1, -1, 0.5, 0.8],
         [-1, -1, -1, 0.7],
         [-1, -1, -1, -1]]
    """
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
              'Mexico', 'Netherlands', 'Denmark', 'Germany', 'Uruguay', 'Switzerland', 'United States', 'Croatia',
              'Senegal', 'Iran', 'Japan', 'Morocco', 'Serbia', 'Poland', 'South Korea', 'Tunisia',
              'Cameroon', 'Canada', 'Ecuador', 'Saudi Arabia', 'Ghana', 'Wales', 'Costa Rica', 'Australia'

              ]
