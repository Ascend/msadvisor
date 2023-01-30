import sys
sys.path.append("..")
from model import evaluate
import time
import os


if __name__ == '__main__':

    os.chdir("../")

    print('################INPUT TEST################')

    data_path = "./test/data"

    #model do not exist
    print("Input parameter:",'{"om_model":"onnx_google.omtest"}')
    evaluate(data_path,'{"om_model":["onnx_google.omtest"]}')
    print("\n")

    #invalid om model file
    print("Input parameter:",'{"om_model":["invalid_model"]}')
    try:
        evaluate(data_path,'{"om_model":["invalid_model"]}')
    except:
        print("error catch")
        print("\n")

    print('\n################OUTPUT & TIME TEST################')
    #find sequence
    print("Input parameter:",'{"om_model":["onnx_google.om"]}')
    start_time = time.time()
    result= evaluate(data_path,'{"om_model":["onnx_google.om"]}')
    end_time = time.time()
    print(result)
    print("time usage:", str(end_time-start_time) + "s")
    print("\n")

    #can not find sequence
    print("Input parameter:",'{"om_model":["onnx_model.om"]}')
    result = evaluate(data_path,'{"om_model":["onnx_model.om"]}')
    print(result)
    print("\n")