import sys
import pandas as pd
import torch
import random
import socket


def load_data(client_num):
    # Import data
    data = pd.read_csv('./FLData/calhousing_train_client{client_num}.csv'.format(client_num))
    # Extract data into X and y
    headers = data.columns
    X_train = data[headers[1:]]
    y_train = data[headers[0]]


    data = pd.read_csv('./FLData/calhousing_test_client{client_num}.csv'.format(client_num))
    # Extract data into X and y
    headers = data.columns
    X_test = data[headers[1:]]
    y_test = data[headers[0]]


    return X_train, y_train, X_test, y_test
    




if __name__ == "__main__":
    # Args
    if len(sys.argv) != 4:
        raise Exception("Invalid number of arguments+")

    client_name = sys.argv[1]
    client_number = int(client_name[len(client_name)-1])
    port_number = int(sys.argv[2])
    method = sys.argv[3]
    host = "127.0.0.1"	
    host_port = 6000



    print("Attempting to connect to host")
    # Create connection to host. 
    connected = False
    while not connected:
        s_send = socket.socket(socket.AF_INET,socket.SOCK_STREAM,)
        s_send.connect((host,host_port))
        # The target port will correspond with the socket that attaches to it. 
        connected = True
    
    print("Connected to host")

    print("Opening Server")

    # Let host connect to client:
    s_rec = socket.socket()
    s_rec.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)  # solution for "[Error 89] Address already in use". Use before bind()
    s_rec.bind((host, port_number))
    s_rec.listen(1) # Number of connections is the number of neighbours in the graph.
    conn, addr = s_rec.accept()


    print("Succesfully connected")



