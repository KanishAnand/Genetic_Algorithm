import json
import requests
import numpy as np
import sys

######### DO NOT CHANGE ANYTHING IN THIS FILE ##################
API_ENDPOINT = 'http://10.4.21.147'
PORT = 3000
MAX_DEG = 11
POPULATION_SIZE = 10
GENERATIONS = 10
TRAIN_RATIO = 0.3
VAL_RATIO = 0.7
TEAM_ID = "MsOYrg4QoHcnSUht1hvbjhYM5BgzBcQT5HO3WVReiC338ykhP1"
# TEAM_ID_D = "hTGuBTgPhst20ZD8eZcFbCa53pWpgghVDSaKNBzn3DE2RDQEuz"
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
    if len(sys.argv) == 2 and sys.argv[1] == "SERVER":
        err = get_errors(TEAM_ID, vector)
        return err
    else:
        return [1.44345, np.random.uniform(0, 10.234235)]

    # print(err)
    # sys.exit()
    # return [1, 3]


def crossover(ind, population):
    c = np.random.randint(1, len(population))
    first_part = population[ind[0]][:c]
    second_part = population[ind[1]][c:]
    children = np.concatenate((first_part, second_part))
    return children


def mutation(children):
    # adding some random number to any 4 elements of children
    ind = np.random.choice(children.shape[0], 4, replace=False)
    # children[9] = children[9] + np.random.uniform(-1*1e-11, 1*1e-11)
    # if(np.random.randint(-1, 1) > 0):
    #     children[9] += 1e-12
    # else:
    #     children[9] += -1e-12
    # children[9] = 4.006171586044872e-11
    # children[9] = min(10, children[9])
    # children[9] = max(-10, children[9])
    # ind = [4, 5]
    for i in ind:
        # children[i] = children[i] + np.random.uniform(-1*1e-13, 1*1e-1)
        val = 0.00001
        if np.random.randint(-1, 1) == 0:
            val = -val
        else:
            pass
        children[i] += children[i]*val
        children[i] = min(10, children[i])
        children[i] = max(-10, children[i])
    return list(children)


def ga():
    # # build up initial population
    # wghts = [0.0, 0.1240317450077846, -6.211941063144333, 0.04933903144709126, 0.03810848157715883, 8.132366097133624e-05, -
    #          6.018769160916912e-05, -1.251585565299179e-07, 3.484096383229681e-08, 4.1614924993407104e-11, -6.732420176902565e-12]

    # population = []
    # indx = []

    # # building initial population
    # for i in range(POPULATION_SIZE):
    #     population.append(wghts.copy())
    #     indx.append(i)

    # # adding noise to make initial population
    # for i in range(POPULATION_SIZE):
    #     # lst = list(np.random.normal(0, 1, 11))
    #     for j in range(len(population[i])):
    #         population[i][j] = population[i][j] + \
    #             np.random.uniform(-1*1e-13, 1*1e-13)
    #         population[i][j] = min(10, population[i][j])
    #         population[i][j] = max(-10, population[i][j])

    # store_population = []

    # for i in range(POPULATION_SIZE):
    #     er = fit(population[i])
    #      val = TRAIN_RATIO*er[0] + VAL_RATIO*er[1]
        # val = er[0]*er[1]*er[1]
    #     store_population.append((val, population[i], er[0], er[1]))

    # use previous best output as initial population
    with open('population.json') as f:
        store_population = json.loads(f.read())

    store_population1 = []

    for pop in store_population:
        er = fit(pop[1])
        val = TRAIN_RATIO*er[0] + VAL_RATIO*er[1]
        store_population1.append((val, pop[1], er[0], er[1]))

    store_population = store_population1

    gen = 0
    while gen < GENERATIONS:
        gen += 1
        probability = []
        error = []
        total = 0

        population = []
        sz = 0
        store_population.sort(key=lambda x: x[0])
        for pop in store_population:
            key = pop[0]
            val = pop[1]
            # print(key, val)
            # print('\n')
            sz += 1
            population.append(val)
            error.append(key)
            total += 1/key
            if sz == POPULATION_SIZE:
                break

        print(store_population[0][2], store_population[0][3])
        print('\n')
        # print("error values : " + str(error))

        # convert errors value to probability
        for er in error:
            # prob = np.exp(-er)/total
            prob = 1/(er*total)
            probability.append(prob)

        # print("probability values : " + str(probability))
        i = 0

        while i < POPULATION_SIZE:
            i += 1
            # choose two parents according to their probability values
            ind = np.random.choice(
                POPULATION_SIZE, 2, replace=False, p=probability)
            # print("index of parents : " + str(ind))

            # crossover of parents to make child
            children = crossover(ind, population)
            # print("children created from crossover: " + str(children))

            # mutation of children
            children = mutation(children)
            # print("children created from mutation: " + str(children))

            # store this children
            er = fit(children)
            val = TRAIN_RATIO*er[0] + VAL_RATIO*er[1]
            # val = er[0]*er[1]*er[1]
            store_population.append((val, children, er[0], er[1]))

    if len(sys.argv) == 2 and sys.argv[1] == "SERVER":
        # saving top 10 fittest members to file
        store_population.sort(key=lambda x: x[0])
        print(store_population[0][2], store_population[0][3])
        print('\n')
        # sz = 0
        # st = ""
        # for pop in store_population:
        #     sz += 1
        #     st += '\"'
        #     st += str(pop[0]) + " " + str(pop[1]) + " " + \
        #         str(pop[2]) + " " + str(pop[3])
        #     st += '\"'
        #     st += '\n'
        #     if sz == POPULATION_SIZE:
        #         break

        with open('population.json', 'w') as f:
            f.write(json.dumps(store_population[:POPULATION_SIZE], indent=4))


if __name__ == "__main__":
    """
    Replace "test" with your secret ID and just run this file
    to verify that the server is working for your ID.
    """

    ga()

    # wghts = [0.0, 0.1240317450077846, -6.211941063144333, 0.04933903144709126, 0.03810848157715883, 8.132366097133624e-05, -
    #          6.018769160916912e-05, -1.251585565299179e-07, 3.484096383229681e-08, 4.1614924993407104e-11, -6.732420176902565e-12]
    # wghts1 = [
    #     -3.2213119791400407e-13,
    #     0.12403174500837291,
    #     -6.2119410631458996,
    #     0.04933903144677353,
    #     0.03810848157722921,
    #     8.132366154783281e-5,
    #     -6.0187691822916674e-5,
    #     -1.2516173291412288e-7,
    #     3.484098931034179e-8,
    #     4.038278175216079e-11,
    #     -6.6990120153011976e-12
    # ]

    # wghts = [
    #     -10.0,
    #     0.2376134120659597,
    #     -6.189375944838398,
    #     0.05308613026201062,
    #     0.03811229204550535,
    #     8.165761073704032e-05,
    #     -6.016362071079147e-05,
    #     -1.2369153905552713e-07,
    #     3.483802358746791e-08,
    #     3.9215234129725944e-11,
    #     -6.701692022074835e-12
    # ]

    # err = get_errors(TEAM_ID, wghts)
    # print(err)
    # assert len(err) == 2

    # submit_status = submit(
    #     TEAM_ID, list(-np.arange(0, 1.1, 0.1)))
    # assert "submitted" in submit_status
