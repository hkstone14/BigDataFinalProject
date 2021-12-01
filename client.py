import socket

if __name__ == "__main__":
    HOST = 'localhost'  # The server's hostname or IP address
    PORT = 9009  # The port used by the server

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((HOST, PORT))
        # s.sendall(b'Hello, world')
        data = s.recv(1024).decode()

    print('Received', repr(data))
