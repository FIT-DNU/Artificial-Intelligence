import random
import numpy as np

# Number of individuals in each generation
POPULATION_SIZE = 500

# Valid genes
GENES = '''abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ 1234567890, .-;:_!"#%&/()=?@${[]}'''

# Target string to be generated
TARGET = "hoilamgi"

class Individual:
    '''
    Class representing individual in population
    '''
    def __init__(self, chromosome):
        self.chromosome = chromosome
        self.fitness = self.cal_fitness()

    @classmethod
    def mutated_genes(cls):
        '''
        Create random genes for mutation
        '''
        return random.choice(GENES)

    @classmethod
    def create_gnome(cls):
        '''
        Create chromosome or string of genes
        '''
        return [cls.mutated_genes() for _ in range(len(TARGET))]

    def mate(self, par2):
        '''
        Perform mating and produce new offspring
        '''
        child_chromosome = []
        for gp1, gp2 in zip(self.chromosome, par2.chromosome):
            prob = random.random()
            if prob < 0.5:
                child_chromosome.append(gp1)
            elif prob < 0.80:
                child_chromosome.append(gp2)
            else:
                child_chromosome.append(self.mutated_genes())
        return Individual(child_chromosome)

    def cal_fitness(self):
        '''
        Calculate fitness score
        '''
        return sum(1 for gs, gt in zip(self.chromosome, TARGET) if gs != gt)

def main():
    global POPULATION_SIZE

    # Initialize population
    population = [Individual(Individual.create_gnome()) for _ in range(POPULATION_SIZE)]

    generation = 1
    # found = False
    found = 0
    while found < 100:
        # Sort population by fitness
        population = sorted(population, key=lambda x: x.fitness)

        # Check if the best individual has fitness 0
        if population[0].fitness == 0:
            found = True
            break

        # Generate new generation
        new_generation = []

        # Elitism: Keep top 10% of the population
        s = int((10 * POPULATION_SIZE) / 100)
        new_generation.extend(population[:s])

        # From 50% of the fittest population, mate to produce offspring
        s = int((90 * POPULATION_SIZE) / 100)
        for _ in range(s):
            parent1 = random.choice(population[:50])
            parent2 = random.choice(population[:50])
            child = parent1.mate(parent2)
            new_generation.append(child)

        population = new_generation

        # Print progress
        print(f"Generation: {generation}\tString: {''.join(population[0].chromosome)}\tFitness: {population[0].fitness}")
        generation += 1
        found += 1

    # Print final result
    if found >= 100:
        print("Khong tim thay loi giai sau 100 lan chay")
    else:
        print(f"Generation: {generation}\tString: {''.join(population[0].chromosome)}\tFitness: {population[0].fitness}")

if __name__ == '__main__':
    main()