import os, json, random
import numpy as np

#-------------FETCHING TRAINING DATA-----------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
td_file_path = os.path.join(SCRIPT_DIR, "training_data.json") #training data path

def get_training_data(td_file_path):
    with open(td_file_path, 'r') as f:
        return json.load(f)
    
data = get_training_data(td_file_path)
games = []

for val in data.values():
    games.append(val)


#----------------------------------------------------------------------------------------------------------


#-------------INITIATION-----------------------------------------------------------------------------------
input_layer = [0,0,0]
hidden_layer = [0,0,0,0,0,0,0,0]
output_layer = [0,0,0]

weights1 = []
weights2 = []

bias_hidden = []
bias_output = []


def initiate_weights(input_layer, hidden_layer, weights1, bias_hidden, bias_output):
    for i in range(len(input_layer)):
        temp = []
        for j in range(len(hidden_layer)):
            temp.append(random.random())
        weights1.append(temp)
    
    for i in range(len(hidden_layer)):
        temp = []
        for j in range(len(output_layer)):
            temp.append(random.random())
        weights2.append(temp)

    for i in range(len(hidden_layer)):
        bias_hidden.append(random.random()/2)
    
    for i in range(len(output_layer)):
        bias_output.append(random.random()/2)

initiate_weights(input_layer, hidden_layer, weights1, bias_hidden, bias_output)
#----------------------------------------------------------------------------------------------------------
param_file = os.path.join(SCRIPT_DIR, "parameters.json") #PARAMETER FILE PATH
def store_param(w_h, w_o, b_h, b_o, param_file, training_completed):
    parameters = {}
    parameters['HIDDEN_LAYER_WEIGHTS'] = w_h 
    parameters['OUTPUT_LAYER_WEIGHTS'] = w_o
    parameters['HIDDEN_LAYER_BIASES'] = b_h
    parameters['OUTPUT_LAYER_BIASES'] = b_o

    try: 
        with open(param_file, 'w') as pf:
            json.dump(parameters, pf, indent=4)
        return True
    except:
        return False

#-----------CORE-LOOP--------------------------------------------------------------------------------------

learning_rate = 0.05

def log_res(condition, argmax, strongest):
    print(f"\n\n-------------\n PLAY: {condition} ; RES: {strongest}, ",end="")
    match argmax:
        case 0:
            print("ROCK \n---------")
        case 1:
            print("PAPER \n---------")
        case 2:
            print("SCISSORS \n---------")

def sigmoid(vector):
    for v in range(len(vector)):
        vector[v] = 1/(1+pow(np.e,-vector[v]))
    return vector

def feed_forward(source_layer, weights, biases):
    layer_inputs = np.dot(source_layer, weights) + biases
    return sigmoid(layer_inputs)

def loss(ideal, output_layer):
    desired_vector = []
    errors = [0,0,0]
    match ideal:
        case 0:
            desired_vector = [1,0,0]
        case 1:
            desired_vector = [0,1,0]
        case 2:
            desired_vector = [0,0,1]

    for i in range(3):  # Simple loop over 3 outputs
        errors[i] = (output_layer[i] - desired_vector[i])

    return errors

def back_prop_output(learning_rate, hidden_layer, output_layer, weights2, error):
    for oi, _ in enumerate(output_layer):
        for hi, h in enumerate(hidden_layer): 
            del_w = learning_rate * h * 2 * error[oi]
            weights2[hi][oi] -= del_w

def back_prop_hidden(learning_rate, input_layer, hidden_layer, output_layer, w1, w2, error):
    total_blame = [0] * len(hidden_layer)
    for hi, h in enumerate(hidden_layer):
        for oi, o in enumerate(output_layer):
            total_blame[hi] += w2[hi][oi] * output_layer[oi] * (1 - output_layer[oi]) * 2* error[oi]
        
    for hi, h in enumerate(hidden_layer):
        hidden_layer_part = h * (1-h)
        for ii, i in enumerate(input_layer):
            del_w = learning_rate * i * hidden_layer_part * total_blame[hi]
            w1[ii][hi] -= del_w
            
    
def core(games, input_layer, hidden_layer, output_layer, weights1, weights2, bias_hidden, bias_output, param_file):
    training_completed = False
    for id, _ in enumerate(games):
        condition = games[id][0]
        #ideal = games[id+1][1] if id != len(games)-1 else games[0][0]
        ideal = games[id+1][1] if id != len(games)-1 else games[0][0]

        match condition:
            case 0:
                input_layer = [1,0,0]
            case 1:
                input_layer = [0,1,0]
            case 2:
                input_layer = [0,0,1]
            case _:
                print("INVALID INPUT?? how did that happen? check the damn Training Data File")
                exit()

        hidden_layer = feed_forward(input_layer, weights1, bias_hidden)

        output_layer = feed_forward(hidden_layer, weights2, bias_output)

        strongest = output_layer[0]
        best_index = 0
        for index, output in enumerate(output_layer):
            if output > strongest:
                strongest = output
                best_index = index

        #log_res(condition, best_index, strongest)

        error = loss(ideal, output_layer)
        print(error)

        back_prop_output(learning_rate, hidden_layer, output_layer, weights2, error)

        back_prop_hidden(learning_rate, input_layer, hidden_layer, output_layer, weights1, weights2, error)

        training_completed = store_param(weights1, weights2, bias_hidden, bias_output, param_file, training_completed)

    if training_completed:
        print("SUCCESSFULLY COMPLETED TRAINING")
    else:
        print("ERROR DURING TRAINING")

#---------------------------------------------------------------------------------------------------------------------------

#--------PLAY TEST----------------------------------------------------------------------------------------------------------

def play_log(best_index):
    ai_play = ""
    match best_index:
        case 0:
            ai_play = "ROCK"
        case 1:
            ai_play = "PAPER"
        case 2:
            ai_play = "SCISSORS"

    print(f"AGENT: {ai_play},", end=" ")

def play(input_layer, hidden_layer, output_layer, weights1, weights2, bias_hidden, bias_output):
    user_play = 'r'
    choices = ['ROCK', 'PAPER', 'SCISSORS']
    print("AGENT: ", random.choice(choices), end=" ")
    while user_play in 'rps':
        user_play = input("INPUT: ")
        match user_play:
            case 'r':
                input_layer = [1,0,0]
            case 'p':
                input_layer = [0,1,0]
            case 's':
                input_layer = [0,0,1]
            case _:
                print("FINISHING GAME")
                exit()

        hidden_layer_input = np.dot(input_layer, weights1) + bias_hidden
        hidden_layer = sigmoid(hidden_layer_input)

        output_layer_input = np.dot(hidden_layer, weights2) + bias_output
        output_layer = sigmoid(output_layer_input)

        strongest = output_layer[0]
        best_index = 0
        
        for index, output in enumerate(output_layer):
            if output > strongest:
                strongest = output
                best_index = index
                

        play_log(best_index)
        
core(games, input_layer, hidden_layer, output_layer, weights1, weights2, bias_hidden, bias_output, param_file)

play(input_layer, hidden_layer, output_layer, weights1, weights2, bias_hidden, bias_output)
            
#--------------------------------------------------------------------------------------------------------