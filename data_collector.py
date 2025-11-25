import os, random, json

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
td_file_path = os.path.join(SCRIPT_DIR, "training_data.json")

record = {}

options = [0,1,2]
literals = ['ROCK','PAPER','SCISSORS']

game_length = 10

def record_data(record, count, opp, user):
    match user:
        case 'r':
            record[count] = [opp, 0]
            return True
        case 'p':
            record[count] = [opp, 1]
            return True
        case 's':
            record[count] = [opp, 2]
            return True
        case _:
            return False
        
def run_loop(record, options, literals):
    game_length = int(input("Enter game length: "))
    for i in range(game_length):
        curr = random.choice(options)

        user = input(f"\ngame {i+1}: \nEnter choice(r/p/s): ")

        print(f"Robot: {literals[curr]}\n")

        record_data(record, (i+1) , curr, user)


def dump(record, td_file_path):
    print("---------------------\n RECORDING GAME\n")
    with open(td_file_path, "w") as file:
        json.dump(record, file, indent=4)
        print("RECORD SUCCESSFUL \n--------------------")

run_loop(record, options, literals)
dump(record, td_file_path)





