import socket

if __name__ == "__main__":
    HOST = 'localhost'  # The server's hostname or IP address
    PORT = 5555  # The port used by the server

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((HOST, PORT))
        td = 1
        while td == 1:
            data = s.recv(1024).decode()
            print('Received data : ', repr(data))

    #print('Received', repr(data))
