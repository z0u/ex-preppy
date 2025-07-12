from typing import override

from proxy.common.flag import flags
from proxy.http.proxy.plugin import HttpProxyBasePlugin


class UpstreamAuthPlugin(HttpProxyBasePlugin):
    flags.add_argument('--upstream-token', type=str, help='The token to use in the Authorization header.')

    @override
    def before_upstream_connection(self, request):
        request.add_header('Authorization'.encode(), f'Bearer {self.flags.upstream_token}'.encode())
        return request
