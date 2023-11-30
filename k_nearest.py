import math
import numpy as np
import pandas as pd

class Sample:
    '''
    A class representing one iris flower of the iris data-set.
    ...
    
    Attributes:
    
    petal_length: float
        The length of the iris petal.
    petal_width: float
        The width of the iris petal.
    sepal_lenght: float
        The length of the iris sepal.
    sepal_width: float
        The width of the iris sepal.
    species: str 
        The species of the iris. Could be Iris virginica, Iris versicolor or Iris setosa.
    magnitude: float
        Represents the magnitude of the vector in the 4-D space defined by length and width of both petal and sepal of a particular sample.        
        For more details on the calculation check https://phys.libretexts.org/Courses/Gettysburg_College/Gettysburg_College_Physics_for_Physics_Majors/03%3A_C3)_Vector_Analysis/3.04%3A_Vector_Algebra_in_Multiple_Dimensions-_Calculations
    '''
    
    def __init__(
        self, 
        petal_length: float, 
        petal_width: float,
        sepal_lenght: float,
        sepal_width: float,
        species: str = 'NA'
        ) -> None:
        '''
        Constructs all the attibutes for the sample object.
        
        Parameters
        __________
        petal_length: float
            The length of the iris petal.
        petal_width: float
            The width of the iris petal.
        sepal_lenght: float
            The length of the iris sepal.
        sepal_width: float
            The width of the iris sepal.
        species: str 
            The species of the iris. Could be Iris virginica, Iris versicolor or Iris setosa.
        '''

        self.petal_length = petal_length
        self.petal_width = petal_width
        self.sepal_lenght = sepal_lenght
        self.sepal_width = sepal_width
        self.species = species
        self.magnitude = math.sqrt(petal_length ** 2 + petal_width ** 2 + sepal_lenght ** 2 + sepal_width ** 2)
    
    def __str__(self):
        return f"Species: {self.species}, Petal length: {self.petal_length}, Petal width: {self.petal_width}, Sepal lenght: {self.sepal_lenght}, Sepal width: {self.sepal_width}, Magnitude: {self.magnitude}"

class Sample_generator:
    '''
    A class to generate random irises samples for testing porpuses. Sample values are taken from a normal distribution.
    
    Methods
    _______
    gen(
        seed: int, 
        number_of_samples: int, 
        petal_length_mean: float, 
        petal_length_std: float, 
        petal_width_mean:float,
        petal_width_std: float,
        sepal_length_mean: float,
        sepal_length_std: float,
        sepal_width_mean: float,
        sepal_width_std: float,
        species: str) -> list[Sample]:
        
        Creates a list containing random iris samples taken from normal distributions. 
    
    '''
    def gen(
        seed: int, 
        number_of_samples: int, 
        petal_length_mean: float, 
        petal_length_std: float, 
        petal_width_mean:float,
        petal_width_std: float,
        sepal_length_mean: float,
        sepal_length_std: float,
        sepal_width_mean: float,
        sepal_width_std: float,
        species: str) -> list[Sample]:
        '''
        Generates a reproducible set of irises samples by seeding a pseudorandom number generator. The generator
        is used to create 4 normal distributions defined by the mean and standard deviation values for petal length,
        petal width, sepal length and sepal width.
        
        Parameters:
        ___________
        seed: int
            Number used for seeding the pseudorandom number generator.
        number_of_samples: int
            Number of samples to be produced.
        petal_length_mean: float
            Mean value for the petal length distribution.
        petal_length_std: float
            Standard deviation for the petal length distribution.
        petal_width_mean:float
            Mean value for the petal width distribution.
        petal_width_std: float
            Standard deviation for the petal width distribution.
        sepal_length_mean: float
            Mean value for the sepal length distribution.
        sepal_length_std: float
            Standard deviation for the sepal length distribution.
        sepal_width_mean: float
            Mean value for the sepal width distribution.
        sepal_width_std: float
            Standard deviation for the sepal width distribution.
        species: str
            The species of the iris. Could be Iris virginica, Iris versicolor or Iris setosa.
        '''

        random_generator = np.random.default_rng(seed=seed)

        samples = []



        for i in range(number_of_samples):
            petal_length = 0
            petal_width = 0
            sepal_length = 0
            sepal_width = 0

            while petal_length <= 0:
                petal_length = random_generator.normal(loc=petal_length_mean, scale=petal_length_std)

            while petal_width <= 0:
                petal_width = random_generator.normal(loc=petal_width_mean, scale=petal_width_std)

            while sepal_length <= 0:  
                sepal_length = random_generator.normal(loc=sepal_length_mean, scale=sepal_length_std)

            while sepal_width <= 0:  
                sepal_width = random_generator.normal(loc=sepal_width_mean, scale=sepal_width_std)

            samples.append(Sample(petal_length, petal_width, sepal_length, sepal_width, species))

        return samples if number_of_samples > 1 else samples[0]
        
class Model:
    """
    Class representing a k-nearest neighbors classificator. It stores locally a copy of the training data both as Sample objects and 
    as magnitudes of vectors. Value of the k constant can be adjusted manually.
    
    Attributes:
    ___________
    
    training: 'list[Sample]'
        List contanining the Sample objects required to train the algorithm
    k: int
        Integer defining the k constant
    vectors: 'list[float]'
        List containing the magnitudes of the vectors representing each sample in petal/sepal space
    labels: set
        Set containing all the species present in the training set.
    
    Methods
    _______
    fit_model(data: 'list[Sample]') -> None
        Trains the k-nearest neighbors algorithm
    
    classify(unknown_sample: Sample) -> str
        Predicts the species of given sample based on the training data
    """
    
    def __init__(self, training: 'list[Sample]', k: int=1):
        self.training = training
        self.k = k
        self.vectors = []
        self.labels = {sample.species for sample in self.training}
    
    
    def __str__(self):
        return f'K-nearest-neightbor model with a trainning set of {len(self.training)} samples and k = {self.k}'
    
    
    def fit_model(self, data: 'list[Sample]') -> None:
        '''
        Populates and sorts the "vectors" attribute by extracting the magnitude of each Sample object in the training set.
        The result is a list with all the magnitudes of the 4-D vectors of the samples sorted from minimum to maximum.
        
        Parameters
        __________
        data: 'list[Sample]'
            Training data for the algorithm
        '''
        self.vectors = [(s.magnitude, s.species) for s in data]
        self.vectors.sort(key=lambda x: x[0])


    #IMPROVEMENT: when having the same number of close neightbors there should be a way to break the tie !!!
    def classify(self, unknown_sample: Sample) -> str:
        '''
        Executes the k-nearest neighbors algorithm to classify an unkown sample.
        
        The list of all the ordered magnitudes of the training dataset is scanned to determine the place of an unknown sample.
        Once the place is found, i, two pointers are set at i and i -1 to calculate the difference between the unkown sample vector and those neighboors.
        The nearest neighbor is added to a list and the corresponding pointer is moded 1 space to the left or the right depending if it was i or i -1.
        The process is repeated until we get the k neighbors of we ran out of samples.
        Species counting is performed on the k neighbors list.
        The species with the biggest member count is returned as the unknown sample's species.
        
        For more details on the algorithm check https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm
        
        Parameters
        __________
        unknown_sample: Sample
            Sample object representing a sample whose species is unkown.
        '''
        
        unknown_magnitude = unknown_sample.magnitude
        neightboors = []
        j = 0
        i = 0
        max_i = len(self.vectors)
        
        # Looking for unkown sample place
        while i < max_i and unknown_magnitude > self.vectors[i][0]:
            i += 1
            if i == max_i:
                break
        
        # Handling edge case where the unkown sample is at the begining 
        if i == 0:
            try:
                for p in range(self.k):
                    neightboors.append(self.vectors[p])
                    j += 1
                
            except Exception as e:
                return f"Minimum: {e}"
            
        #Handling edge case where the unkown sample is at end of the list
        elif i == max_i:
            try:
                for p in range(1, self.k + 1):
                    neightboors.append(self.vectors[-p])
                    j += 1
            except Exception as e:
                return f"Maximum: {e}"
            
        # Regular case 
        else:  
            r = i
            l = i - 1
            upper_limit = len(self.vectors)

            # Looping theough
            while j < self.k and r < upper_limit and l >= 0:

                right_diff = abs(unknown_magnitude - self.vectors[r][0])
                left_diff = abs(unknown_magnitude - self.vectors[l][0])

                if right_diff > left_diff:
                    neightboors.append(self.vectors[r])
                    r += 1
                    j += 1
                else:
                    neightboors.append(self.vectors[l])
                    l -= 1
                    j += 1
        
        if j == self.k:
            counter = {label:0 for label in self.labels}

            for member in neightboors:
                counter[member[1]] += 1
                
            max_label = ''
            max_value = 0
            keys = list(counter.keys())
            
        
            for i in range(0, len(keys)):
                if max_value < counter[keys[i]]:
                    max_label = keys[i]
                    max_value = counter[keys[i]]
    
           
            
            return max_label
        else:
            return "Not enough neighbors"

class Engine:
    '''
    Class encapsulating a Model instance. It provides an interface to load data from a csv file and to perform batch testing/prediction.
    
    Attibutes
    _________
    model: Model
        Model instance used by the engine.
        
    Methods
    _______
    _load_data(self, path: str, type_of_sample: str) -> 'list[Sample]'
        Private method used internally to transform a csv file into a list of Samples.
        
    create_model(self, model: Model, path, k: int) -> None
        Creates and trains a Model instance.
    
    predict(self, unknown_data) -> 'list[Sample]
        Classifies a list of unkown samples.
    
    test(self, testing_data) -> float
        Tests the performance of the model by checking its predictions against a known dataset.
    
    ''' 
    
    def __init__(self) -> None:
        self.model = None
        
    
    def _load_data(self, path: str, type_of_sample: str) -> 'list[Sample]':
        '''
        Loads a csv file into a Pandas dataframe to create the list of Sample objects required for training.
        
        Parameters
        __________
        path: str
            Path to the csv file. The csv file should have the columns Id, SepalLengthCm, SepalWidthCm, PetalLengthCm, PetalWidthCm
            and Species in that order. Other wise adjust the indexes of this method.
            
        type_of_sample: str
            Type of sample "known" or "unkown"
        
        '''
        iris = pd.read_csv(path)
        processed_data = []

        for i in range(len(iris)):
            # Adjust indexes according to dataframe structure.
            processed_data.append(Sample(iris.iloc[i][3], iris.iloc[i][4], iris.iloc[i][1], iris.iloc[i][2], iris.iloc[i][5]))
        
        return processed_data
        
    
    def create_model(self, model: Model, path: str, k: int) -> None:
        '''
        Creates Model instance and trains it with the processed data.
        
        Parameters
        __________
        model: Model
            Model instance used for sample classification.
            
        path: str
            Path to the csv file.
            
        k: int
            Number of neighboors to be used in the classification algorithm.
        '''
        training_data = self._load_data(path, type_of_sample='known')
        self.model = Model(training=training_data, k=k)
        self.model.fit_model(self.model.training)
    

    def predict(self, unknown_data: 'list[Sample]') -> 'list[Sample]':
        '''
        Sends a list of unknown samples (samples with species = 'NA') to the Model instance and returns a copy with each sample's species attribute
        updated to the model prediction.
        
        Parameters
        __________
        unknown_data: 'list[Sample]'
            List of Samples objects of unknown species.
        '''
        data = unknown_data.copy()
        
        for sample in data:
            sample.species = self.model.classify(sample)
        
        return data
    
    
    def test(self, testing_data:str) -> float:
        '''
        Tests Model instance peformance against a set of known Samples. Returns accuracy as a float between 0 and 1.
        
        Parameters
        __________
        testing_data_path: str or list[Sample]
            Path to csv file containing testing data. Alternatively could be a list of unknown Samples.
        '''
        
        if isinstance(testing_data, str):
            testing_data = self._load_data(testing_data, type_of_sample='known')
            
        size = len(testing_data)
        successes = 0
        
        for sample in testing_data:
            if self.model.classify(sample) == sample.species:
                successes += 1

        return successes / size

if __name__ == "__main__":
  '''
  Creation of dataset suitable for testing
  '''
  
  # Replace paths as needed
  
  df = pd.read_csv('/kaggle/input/iris-dataset-extended/iris_extended.csv')
  
  test_df = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width','species']]
  test_df['species'].replace(['setosa', 'versicolor', 'virginica'], ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'], inplace=True)
  
  print(test_df)
  
  test_df.to_csv('/kaggle/working/test.csv')

  
  '''
  Testing Engine class against randomly generated data.
  '''
  
  my_engine = Engine()
  
  my_engine.create_model(Model, '/kaggle/input/iris/Iris.csv', 1)
  
  samples_virginica = Sample_generator.gen(1000,10, 5.552, 0.551895, 2.026, 0.27465, 6.588, 0.63588, 2.974, 0.322497, 'Iris-virginica')
  samples_versicolor = Sample_generator.gen(2000,10, 4.26, 0.469911, 1.326, 0.197753, 5.936, 0.516171, 2.77, 0.313798, 'Iris-versicolor')
  samples_setosa = Sample_generator.gen(3000,10, 1.464, 0.173511, 0.244, 0.10721, 5.006, 0.35249, 3.418, 0.381024 ,'Iris-setosa')
  
  samples = []
  
  samples.extend(samples_virginica)
  samples.extend(samples_setosa)
  samples.extend(samples_versicolor)
  
  prediction = my_engine.predict(samples)
  
  
  ''' 
  Testing Engine class against real dataset
  '''
  
  # Replace paths as needed
  
  print(my_engine.test(samples))
  print(my_engine.test('/kaggle/working/test.csv'))
  x=my_engine._load_data('/kaggle/working/test.csv', 's')
  print(x[0])
