# Example usage
from proxy import sleep_loop
from track.proxy.auth_proxy import start_proxy


proxy = start_proxy('http://127.0.0.1:43800', 'your_auth_token')
print(f'Proxy running at {proxy.listen_addr}')
sleep_loop(proxy.proxy)
