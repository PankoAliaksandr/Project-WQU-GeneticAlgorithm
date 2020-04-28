# Miniproject 3 Predicting a Stock Price using a Genetic Algorithm

# Libraries
import pandas_datareader.data as web
import datetime
import random
import numpy as np
import operator
import statsmodels.api as sm
import csv

class GeneticAlgorithm:
    
    def __init__(self):
        # Download Tesla stock prices
        start_date = datetime.date(2014, 2, 18) # first date tesla stock >= 200
        end_date = datetime.date(2017, 9, 1)
        data_frame = web.DataReader('TSLA', 'google', start_date, end_date)
        self.tesla_price = data_frame.Close

        # Initialize the population with a random set of conditions
        self.conditions = []
        
        self._generate_conditions_()
        self._execute_mutation_()
        self._calculate_fitness_function_()

    def _generate_conditions_(self):
        # Determine random value borders
        tesla_price_min = int(min(self.tesla_price))
        tesla_price_max = int(max(self.tesla_price))
        
        # Generate 100 different initial conditions (individuals)
        for i in range(0,100):
            # Each condition (individual) based on 3 intervals. 5 points are used to create 3 intervals
            a = random.randint(tesla_price_min, tesla_price_max)
            b = random.randint(tesla_price_min, tesla_price_max)
            c = random.randint(tesla_price_min, tesla_price_max)
            d = random.randint(tesla_price_min, tesla_price_max)
            e = random.randint(tesla_price_min, tesla_price_max)
            
            # Interval 1: (values >= a)
            # Interval 2: [b;c]
            # Interval 3: [d;e]
            
            # According to chosen intervals borders: b <= c, d <= e
            if b > c:
                temp = b
                b = c
                c =  temp
                
            if d > e:
                temp = d
                d = e
                e =  temp
            # 5 points define an individual   
            individual = (a,b,c,d,e)
            # Add the individual to the conditions list
            self.conditions.append(individual)
    
            # Calculate the standard deviation of tesla prices for the whole period
            self.std_o = np.std(self.tesla_price)  
            
    def _execute_mutation_(self):
        
        conditions_number = len(self.conditions)
        prices_number = len(self.tesla_price)        
        # Use 10 generations (10 mutations)
        for i in range (10):
            # Collect stock prices that pass the series of conditions
            passed_prices_list = []
            self.conditions_prices_dictionary = {}
            
            # Every condition should be examinated
            for y in range(conditions_number):
                # All prices sequences (lengh 3) should be checked
                for x in range(1, prices_number - 1):
                        # Interval 1: (values >= a)
                        # Interval 2: [b;c]
                        # Interval 3: [d;e]
                    if self.tesla_price[x-1] >= self.conditions[y][0] and \
                       self.tesla_price[x]   >= self.conditions[y][1] and \
                       self.tesla_price[x]   <= self.conditions[y][2] and \
                       self.tesla_price[x+1] >= self.conditions[y][3] and \
                       self.tesla_price[x+1] <= self.conditions[y][4]:
                        
                        # If the price sequence fits the condition save it   
                        passed_prices_list.append(self.tesla_price[x])
                
                # Save middle price from the passed prices sequence linked with the condition        
                self.conditions_prices_dictionary[y] = passed_prices_list
                # Prepare passed-prices list for a new condition 
                passed_prices_list = []
               
            # Calculate the fitness function value for every individual      
            fitness_func_dict = {}
            
            # For all prices that passed conditions calculate st.dev.
            for y in range(conditions_number):
                # Take passed prices list by key value of a condition
                prices = self.conditions_prices_dictionary.get(y)
                std = np.std(prices)
                
                # Calculate the fitness function for all conditions
                if len(prices) != 0 and (std/self.std_o)!=0:
                    function = -np.log2(std/self.std_o)-0.1/len(prices)
                    fitness_func_dict[y] = function
            
            # Sort individuals by fitness function value
            self.sorted_fitness_func = sorted(fitness_func_dict.items(), key = operator.itemgetter(1))
            
            # Find the index of 2 weakest individuals in terms of fitness function
            weak_C1_index = self.sorted_fitness_func[0][0]
            weak_C2_index = self.sorted_fitness_func[1][0]
            
            # Use shift approach.
            C1_new = (self.conditions[weak_C1_index][0],self.conditions[weak_C1_index][1], self.conditions[weak_C1_index][2], self.conditions[weak_C1_index][3] + 2, self.conditions[weak_C1_index][4] + 2 )
            C2_new = (self.conditions[weak_C2_index][0],self.conditions[weak_C2_index][1], self.conditions[weak_C2_index][2], self.conditions[weak_C2_index][3] + 2, self.conditions[weak_C2_index][4] + 2 )
            
            self.conditions[weak_C1_index] = C1_new
            self.conditions[weak_C2_index] = C2_new

    def _calculate_fitness_function_(self):
        # All mutations have been done.    
        # Calculate the fitness function value for every individual      
        fitness_func_dict = {}
        for y in range(len(self.conditions_prices_dictionary)):
            prices = self.conditions_prices_dictionary.get(y)
            std = np.std(prices)
            if len(prices) != 0 and (std/self.std_o)!=0:
                function = -np.log2(std/self.std_o)-0.1/len(prices)
                fitness_func_dict[y] = function
        
        # Sort individuals by fitness function value
        self.sorted_fitness_func = sorted(fitness_func_dict.items(), key=operator.itemgetter(1))    
    
    
    # Use linear regression to fit the data extracted for the best individual 
    def forecast_price(self):
        # Find the best individual
        y = list(self.conditions_prices_dictionary[self.sorted_fitness_func[-1][0]])
        # Add constant vector to the model
        # Form the vector for independent variable
        x = list(range(1,len(y) + 1))
        x = sm.add_constant(x)
        model = sm.OLS(y, x)
        result = model.fit()
        
        # Use the model to forecast the next two years (251*2 = 502 days)
        x_extended = range(len(y),len(y) + 502)
        self.predictions = list(result.predict(sm.add_constant(x_extended)))
    
    # Write data to file
    def write_to_file(self, output_file_name):
        
        with open(output_file_name, "w") as file:
            writer = csv.writer(file, delimiter='\n')
            writer.writerow(self.predictions)


gen_alg = GeneticAlgorithm()
gen_alg.forecast_price() 
gen_alg.write_to_file("D:\\output_i.csv")
 






