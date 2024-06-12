from includes import *

# Create an instance of the loader to keep the CSV loaded
csv_loader = CSVLoader('simple_tqc.csv')

def serializeSpaceGroup(num):
    if 1<=num<=1:
        return 0
    elif 2<=num<=2:
        return 1
    elif 3<=num<=5:
        return 2
    elif 6<=num<=9:
        return 3
    elif 10<=num<=15:
        return 4
    elif 16<=num<=24:
        return 5
    elif 25<=num<=46:
        return 6
    elif 47<=num<=74:
        return 7
    elif 75<=num<=80:
        return 8
    elif 81<=num<=82:
        return 9
    elif 83<=num<=88:
        return 10
    elif 89<=num<=98:
        return 11
    elif 99<=num<=110:
        return 12
    elif 111<=num<=122:
        return 13
    elif 123<=num<=142:
        return 14
    elif 143<=num<=146:
        return 15
    elif 147<=num<=148:
        return 16
    elif 149<=num<=155:
        return 17
    elif 156<=num<=161:
        return 18
    elif 162<=num<=167:
        return 19
    elif 168<=num<=173:
        return 20
    elif 174<=num<=174:
        return 21
    elif 175<=num<=176:
        return 22
    elif 177<=num<=182:
        return 23
    elif 183<=num<=186:
        return 24
    elif 187<=num<=190:
        return 25
    elif 191<=num<=194:
        return 26
    elif 195<=num<=199:
        return 27
    elif 200<=num<=206:
        return 28
    elif 207<=num<=214:
        return 29
    elif 215<=num<=220:
        return 30
    elif 221<=num<=230:
        return 31

def classifier():
    return "classification"

# Functions to be used in another module
def valid_ids():
    return csv_loader.valid_ids()

def info(id):
    temp = csv_loader.info(id)
    return jax.nn.one_hot(serializeSpaceGroup(int(temp)), 32)