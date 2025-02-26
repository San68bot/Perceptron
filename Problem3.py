import random
import os

FILE_PATH = "twoclassData/"
NUM_FEATURES = 2

def initialize_weight_vector(length):
    # generate and return random weights
    return [random.uniform(-1, 1) for _ in range(length)]

def train_on_file(weights, filename):
    # open file
    with open(os.path.join(FILE_PATH, filename)) as f:
        # extract lines from file
        examples = f.readlines()

        # loop through lines in file
        for e in examples:
            # unpack values from line 
            x, y, c = tuple(e.strip().split(" "))

            # convert to numeric values
            x = float(x)
            y = float(y)
            c = float(c)
            
            # perform classification
            output = weights[0] * x + weights[1] * y

            # correct weights
            if output <= 0 and c == 1:
                weights[0] += x
                weights[1] += y
            elif output > 0 and c == 0:
                weights[0] -= x
                weights[1] -= y

    # close file
    f.close()

    # return trained weights
    return weights

def test_weights(weights, filename):
    # open file
    with open(os.path.join(FILE_PATH, filename)) as f:
        # extract lines from file
        examples = f.readlines()

        # initialize counts
        correct = 0
        total = 0

        # loop through lines in file
        for e in examples:
            # unpack values from line 
            x, y, c = tuple(e.strip().split(" "))

            # convert to numeric values
            x = float(x)
            y = float(y)
            c = float(c)
            
            # perform classification
            output = weights[0] * x + weights[1] * y

            # check for accuracy
            if (output > 0 and c == 1) or (output <= 0 and c == 0):
                correct += 1
            total += 1
    
    # close file
    f.close()

    # return accuracy rate
    return correct / total


def main():
    weights = initialize_weight_vector(NUM_FEATURES)

    for i in range(1, 11):
        weights = train_on_file(weights, ("set" + str(i) + ".train"))
        print("TRAINED ON SET:", i)
        print("CURRENT WEIGHTS:", weights)
        print("CURRENT ACCURACY:", test_weights(weights, "set.test"))


    print("\nFINAL WEIGHTS:", weights)

    print("FINAL ACCURACY", test_weights(weights, "set.test"))

if __name__ == "__main__":
    main()