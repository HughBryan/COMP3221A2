import sys
import pandas as pd
import torch
import random
import socket
import timeit
import select


def establish_connections(send_sockets,rec_sockets,port_server,host):
    print("Open server")
    # Let clients connect.
    s_send = socket.socket()
    s_send.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)  # solution for "[Error 89] Address already in use". Use before bind()
    s_send.bind((host, port_server))
    s_send.listen(5) # Number of connections is the number of neighbours in the graph.
    send_sockets = [s_send]
    s_send.setblocking(0)

    
    start_time = timeit.default_timer()
    curr_time = start_time
    timeout_time = 30 # 30 seconds


    while curr_time - start_time < timeout_time:
        readable, writable, errored = select.select(send_sockets, [], [],1)
        for s in readable:
            if s is s_send:
                client_socket, address = s_send.accept()
                send_sockets.append(client_socket)
                print ("Connection from", address)

        # As soon as we have more than the starting sersver socket, we beging the count-down for other clients to connect.
        if len(send_sockets) > 1:
            curr_time = timeit.default_timer()                
            print(curr_time - start_time)

    print("All clients connected to server")




if __name__ == "__main__":
    random.seed(1)

    # Args
    if len(sys.argv) != 3:
        raise Exception("Invalid number of arguments+")

    port_server = int(sys.argv[1])
    sub_client = sys.argv[2]
    host = "127.0.0.1"
    send_sockets = []
    rec_sockets = []

    establish_connections(send_sockets,rec_sockets,port_server,host)


    # Ready to start initalising and sharing data. 