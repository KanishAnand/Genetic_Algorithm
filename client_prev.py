import json
import requests
import numpy as np
import sys

######### DO NOT CHANGE ANYTHING IN THIS FILE ##################
API_ENDPOINT = 'http://10.4.21.147'
PORT = 3000
MAX_DEG = 11
POPULATION_SIZE = 10
GENERATIONS = 2
TEAM_ID = "MsOYrg4QoHcnSUht1hvbjhYM5BgzBcQT5HO3WVReiC338ykhP1"
# functions that you can call


def get_errors(id, vector):
    """
    returns python array of length 2 
    (train error and validation error)
    """
    for i in vector:
        assert -10 <= abs(i) <= 10
    assert len(vector) == MAX_DEG

    return json.loads(send_request(id, vector, 'geterrors'))


def submit(id, vector):
    """
    used to make official submission of your weight vector
    returns string "successfully submitted" if properly submitted.
    """
    for i in vector:
        assert -10 <= abs(i) <= 10
    assert len(vector) == MAX_DEG
    return send_request(id, vector, 'submit')

# utility functions


def urljoin(root, port, path=''):
    root = root + ':' + str(port)
    if path:
        root = '/'.join([root.rstrip('/'), path.rstrip('/')])
    return root


def send_request(id, vector, path):
    api = urljoin(API_ENDPOINT, PORT, path)
    vector = json.dumps(vector)
    response = requests.post(api, data={'id': id, 'vector': vector}).text
    if "reported" in response:
        print(response)
        exit()

    return response


def fit(vector):
    # return training and validation error
    # err = get_errors(TEAM_ID, list(vector))
    # print(err)
    # sys.exit()
    # return err
    return [1, 3]
    # return np.random.randint(1, 3)


def crossover(ind, population):
    c = np.random.randint(1, population.shape[1])
    first_part = population[ind[0]][:c]
    second_part = population[ind[1]][c:]
    children = np.concatenate((first_part, second_part))
    return children


def mutation(children):
    # adding some random number to any 4 elements of children
    ind = np.random.choice(children.shape[0], 4, replace=False)
    for i in ind:
        children[i] += np.random.uniform(-0.02, 0.02)
        children[i] = min(10, children[i])
        children[i] = max(-10, children[i])
    return children


def ga():
    wghts = [0.0, 0.1240317450077846, -6.211941063144333, 0.04933903144709126, 0.03810848157715883, 8.132366097133624e-05, -
             6.018769160916912e-05, -1.251585565299179e-07, 3.484096383229681e-08, 4.1614924993407104e-11, -6.732420176902565e-12]

    population = []
    indx = []

    # building initial population
    for i in range(POPULATION_SIZE):
        population.append(wghts.copy())
        indx.append(i)

    # adding noise to make initial population
    for i in range(POPULATION_SIZE):
        # lst = list(np.random.normal(0, 1, 11))
        for j in range(len(population[i])):
            population[i][j] = population[i][j] + \
                np.random.uniform(-1*1e-13, 1*1e-13)
            population[i][j] = min(10, population[i][j])
            population[i][j] = max(-10, population[i][j])

    population = np.array(population)

    no = 0
    while no < GENERATIONS:
        no += 1
        probability = []
        error = []
        total = 0

        # calculating fittness values
        st = []
        mnn = 1e40
        for val in population:
            er = fit(val)
            if er[1] < mnn:
                mnn = er[1]
                st = er
            error.append(er[1])
            # total += np.exp(-er)
            total += 1/er[1]

        print(st)
        print('\n')
        # print("error values : " + str(error))

        # convert errors value to probability
        for er in error:
            # prob = np.exp(-er)/total
            prob = 1/(er*total)
            probability.append(prob)

        # print("probability values : " + str(probability))
        new_population = []
        i = 0

        while i < POPULATION_SIZE:
            i += 1
            # choose two parents according to their probability values
            ind = np.random.choice(indx, 2, replace=False, p=probability)
            # print("index of parents : " + str(ind))

            # crossover of parents to make child
            children = crossover(ind, population)
            # print("children created from crossover: " + str(children))

            # mutation of children
            children = mutation(children)

            new_population.append(children)

        new_population = np.array(new_population)
        population = new_population.copy()

    mn = 1e40
    minpop = []
    str = []
    for val in population:
        er = fit(val)
        if er[1] < mn:
            mn = er[1]
            minpop = val
            str = er

    print('\n')
    print(minpop)
    print(str)


if __name__ == "__main__":
    """
    Replace "test" with your secret ID and just run this file 
    to verify that the server is working for your ID.
    """
    ga()

    # wghts = [0.0, 0.1240317450077846, -6.211941063144333, 0.04933903144709126, 0.03810848157715883, 8.132366097133624e-05, -
    #          6.018769160916912e-05, -1.251585565299179e-07, 3.484096383229681e-08, 4.1614924993407104e-11, -6.732420176902565e-12]
    # err = get_errors(TEAM_ID, wghts)
    # print(err)
    # assert len(err) == 2

    # submit_status = submit(
    #     TEAM_ID, list(-np.arange(0, 1.1, 0.1)))
    # assert "submitted" in submit_status
