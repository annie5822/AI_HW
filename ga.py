import numpy as np
import random
from typing import List, Tuple, Optional

class GeneticAlgorithm:
    def __init__(
        self,
        pop_size: int,      # Population size
        generations: int,   # Number of generations for the algorithm
        mutation_rate: float,  # Gene mutation rate
        crossover_rate: float,  # Gene crossover rate
        tournament_size: int,  # Tournament size for selection
        elitism: bool,         # Whether to apply elitism strategy
        random_seed: Optional[int],  # Random seed for reproducibility
    ):
        # Students need to set the algorithm parameters
        self.pop_size = pop_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.tournament_size = tournament_size
        self.elitism = elitism

        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)

    def _init_population(self, M: int, N: int) -> List[List[int]]:
        """
        Initialize the population and generate random individuals, ensuring that every student is assigned at least one task.
        :param M: Number of students
        :param N: Number of tasks
        :return: Initialized population
        """
        # TODO: Initialize individuals based on the number of students M and number of tasks N
        population = []
        # 生成 pop_size 個個體
        for j in range(self.pop_size):
            individual = [None] * N  # 建一個大小為 N 的列表來表示任務分配
            tasks = list(range(N))  # 任務的索引從 0 到 N-1
            
            # 確保每個學生至少分配到一個任務
            random.shuffle(tasks)  # 為每個學生隨機分配一個任務
            
            for i in range(M):
                task = tasks.pop()  # 隨機取出一個任務
                individual[task] = i  # 分配該任務給學生 i
      
            # 將剩餘的任務隨機分配給任意學生
            for task in tasks:
                student = random.randint(0, M-1)  # 隨機選擇一個學生
                individual[task] = student  # 分配任務給該學生
            
            # 將這個個體添加到population
            population.append(individual)
        
        return population


    def _fitness(self, individual: List[int], student_times: np.ndarray) -> float:
        """
        Fitness function: calculate the fitness value of an individual.
        :param individual: Individual
        :param student_times: Time required for each student to complete each task
        :return: Fitness value
        """
        # TODO: Design a fitness function to compute the fitness value of the allocation plan
        fitness_score = 0.0  # 初始化score為 0

        # 累加該學生完成所有分配給他的任務
        for task, student in enumerate(individual):
            fitness_score += student_times[student][task]
    
        return fitness_score  # 返回fitness_score

    def _selection(self, population: List[List[int]], fitness_scores: List[float]) -> List[int]:
        """
        Use tournament selection to choose parents for crossover.
        :param population: Current population
        :param fitness_scores: Fitness scores for each individual
        :return: Selected parent
        """
        # TODO: Use tournament selection to choose parents based on fitness scores
        selected_individuals = random.sample(list(enumerate(population)), self.tournament_size)
    
        # 找到這些個體中fitness_score最好的個體
        best_individual = None
        best_fitness = float('inf')  # 初始化

        for index, individual in selected_individuals:
            if fitness_scores[index] < best_fitness:
                best_fitness = fitness_scores[index]
                best_individual = individual
                
        return best_individual

    def _crossover(self, parent1: List[int], parent2: List[int], M: int) -> Tuple[List[int], List[int]]:
        """
        Crossover: generate two offspring from two parents.
        :param parent1: Parent 1
        :param parent2: Parent 2
        :param M: Number of students
        :return: Generated offspring
        """
        # TODO: Complete the crossover operation to generate two offspring
        cr_rate = random.random()
        if  cr_rate > self.crossover_rate:  # 檢查是否進行交叉
            crossover_point = random.randint(0, len(parent1) - 1)  # 隨機選擇一個交叉點
            child1 = parent1[:crossover_point] + parent2[crossover_point:]
            child2 = parent2[:crossover_point] + parent1[crossover_point:]
        
            # 確保每個學生都有任務
            def fix(individual: List[int]) -> List[int]:
                task_count = [0] * M  # 記錄每個學生獲得的任務數
        
                # 統計每個學生的任務數
                for task in individual:
                    task_count[task] += 1
        
                # 找到未被分配任務的學生
                unassigned_students = [i for i in range(M) if task_count[i] == 0]
        
                # 找到被分配了多個任務的學生
                for i in range(len(individual)):
                    if task_count[individual[i]] > 1:
                        # 隨機分配未被分配任務的學生
                        if unassigned_students:
                            task_count[individual[i]] -= 1  # 這個學生的任務數減一
                            new_student = unassigned_students.pop()
                            individual[i] = new_student  # 將這個任務分配給未分配的學生
                            task_count[new_student] += 1  # 新分配的學生任務數加一
        
                return individual
        
            child1 = fix(child1)
            child2 = fix(child2)
        
        else:
            child1, child2 = parent1.copy(), parent2.copy()
        
        return child1, child2
        

    def _mutate(self, individual: List[int], M: int) -> List[int]:
        """
        Mutation operation: randomly change some genes (task assignments) of the individual.
        :param individual: Individual
        :param M: Number of students
        :return: Mutated individual
        """
        # TODO: Implement the mutation operation to randomly modify genes
        mutate_point = random.randint(1, len(individual) - 1)  # 隨機選擇一個交叉點
        mu_rate = random.random()
        if  mu_rate > self.mutation_rate:  # 根據變異率判斷是否進行變異
            # 將前 n 個元素移動到後方
            individual = individual[mutate_point:] + individual[:mutate_point]
        
        return individual
            
    
    def __call__(self, M: int, N: int, student_times: np.ndarray) -> Tuple[List[int], int]:
        """
        Execute the genetic algorithm and return the optimal solution (allocation plan) and its total time cost.
        :param M: Number of students
        :param N: Number of tasks
        :param student_times: Time required for each student to complete each task
        :return: Optimal allocation plan and total time cost
        """
        # TODO: Complete the genetic algorithm process, including initialization, selection, crossover, mutation, and elitism strategy
        # 1. 初始化population
        population = self._init_population(M, N)
    
        # 2. 計算初始population的fitness_score
        fitness_scores = [self._fitness(individual, student_times) for individual in population]
    
        for generation in range(self.generations):
            new_population = []
    
            # 3. 演化過程：生成新一代
            if self.elitism:  # 保留最優個體到下一代
                best_individual = population[fitness_scores.index(min(fitness_scores))]
                new_population.append(best_individual)
    
            while self.pop_size > len(new_population):
                # 選擇父代
                parent1 = self._selection(population, fitness_scores)
                parent2 = self._selection(population, fitness_scores)
    
                # 交叉生成後代
                child1, child2 = self._crossover(parent1, parent2, M)
    
                # 變異操作
                child1 = self._mutate(child1, M)
                child2 = self._mutate(child2, M)
    
                # 加後代到新種群
                new_population.append(child1)
                if  self.pop_size > len(new_population):
                    new_population.append(child2)
    
            # 更新 population
            population = new_population
    
            # 4. 計算 new_population 的 fitness_score
            fitness_scores = [self._fitness(individual, student_times) for individual in population]
    
        # 5. 找出最優的個體和結果
        best_fitness = int(min(fitness_scores))
        best_individual = population[fitness_scores.index(best_fitness)]
    
        return best_individual, best_fitness


if __name__ == "__main__":
    def write_output_to_file(problem_num: int, total_time: int, filename: str = "results.txt") -> None:
        """
        Write results to a file and check if the format is correct.
        """
        print(f"Problem {problem_num}: Total time = {total_time}")

        if not isinstance(total_time, int) :
            raise ValueError(f"Invalid format for problem {problem_num}. Total time should be an integer.")
        
        with open(filename, 'a') as file:
            file.write(f"Total time = {total_time}\n")

    # TODO: Define multiple test problems based on the examples and solve them using the genetic algorithm
    # Example problem 1 (define multiple problems based on the given example format)
    # M, N = 2, 3
    # student_times = [[3, 8, 6],
    #                  [5, 2, 7]]

    M1, N1 = 2, 3
    cost1 = [[3,2,4],[4,3,2]]
   
    M2, N2 = 4, 4
    cost2 =  [[5,6,7,4],[4,5,6,3],[6,4,5,2],[3,2,4,5]]
    
    M3, N3 = 8, 9
    cost3 = [[90, 100, 60, 5, 50, 1, 100, 80, 70],[100, 5, 90, 100, 50, 70, 60, 90, 100],[50, 1, 100, 70, 90, 60, 80, 100, 4],
             [60, 100, 1, 80, 70, 90, 100, 50, 100],[70, 90, 50, 100, 100, 4, 1, 60, 80],[100, 60, 100, 90, 80, 5, 70, 100, 50],
             [100, 4, 80, 100, 90, 70, 50, 1, 60],[1, 90, 100, 50, 60, 80, 100, 70, 5]]
    
    M4, N4 = 3, 3
    cost4 = [[2,5,6],[4,3,5],[5,6,2]]
    
    M5, N5 = 4, 4
    cost5 = [[4,5,1,6],[9,1,2,6],[6,9,3,5],[2,4,5,2]]
    
    M6, N6 = 4, 4
    cost6 = [[5,4,6,7],[8,3,4,6],[6,7,3,8],[7,8,9,2]]
    
    M7, N7 = 4, 4   #這題是小數(我用整數可能要改)
    cost7 = [[4,7,8,9],[6,3,6,7],[8,6,2,6],[7,8,7,3]]
    
    M8, N8 = 5, 5
    cost8 = [[8,8,24,24,24],[6,18,6,18,18],[30,10,30,10,30],[21,21,21,7,7],[27,27,9,27,9]]
    
    M9, N9 = 5, 5 ###
    cost9 = [[10,10,10000,10000,10000],
             [12,10000,10000,12,12],
             [10000,15,15,10000,10000],
             [11,10000,11,10000,10000],
             [10000,14,10000,14,14]]
    
    M10, N10 = 9, 10
    cost10 =  [[1, 90, 100, 50, 70, 20, 100, 60, 80, 90],[100, 10, 1, 100, 60, 80, 70, 100, 50, 90],
               [90, 50, 70, 1, 100, 100, 60, 90, 80, 100],[70, 100, 90, 5, 10, 60, 100, 80, 90, 50],
               [50, 100, 100, 90, 20, 4, 80, 70, 60, 100],[100, 5, 80, 70, 90, 100, 4, 50, 1, 60],
               [90, 60, 50, 4, 100, 90, 100, 5, 10, 80],[100, 70, 90, 100, 4, 60, 1, 90, 100, 5],
               [80, 100, 5, 60, 50, 90, 70, 100, 4, 1]]

    problems = [(M1, N1, np.array(cost1)),
               (M2, N2, np.array(cost2)),
               (M3, N3, np.array(cost3)),
               (M4, N4, np.array(cost4)),
               (M5, N5, np.array(cost5)),
               (M6, N6, np.array(cost6)),
               (M7, N7, np.array(cost7)),
               (M8, N8, np.array(cost8)),
               (M9, N9, np.array(cost9)),
               (M10, N10, np.array(cost10))]
    
        # Example for GA execution:
        # TODO: Please set the parameters for the genetic algorithm
    ga = GeneticAlgorithm(
            pop_size = 2000,
            generations = 12,
            mutation_rate = 0.2,
            crossover_rate = 0.2,
            tournament_size = 500,
            elitism = True,
            random_seed = 187
    )
    
        # Solve each problem and immediately write the results to the file
    for i, (M, N, student_times) in enumerate(problems, 1):  #分別匯入上面的problems
        best_allocation, total_time = ga.__call__(M=M, N=N, student_times=student_times)
        write_output_to_file(i, total_time)
    
    print("Results have been written to results.txt")
