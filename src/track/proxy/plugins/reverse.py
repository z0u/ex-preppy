from typing import List, Union, Tuple
from proxy.http.server import ReverseProxyBasePlugin
from proxy.common.flag import flags


class ReverseProxyPlugin(ReverseProxyBasePlugin):
    flags.add_argument('--target-addr', type=str, help='The URL stem of the upstream server.')

    def routes(self) -> List[Union[str, Tuple[str, List[bytes]]]]:
        return [
            (r'.*', [self.flags.target_addr.encode()]),
        ]
