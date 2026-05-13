import logging
import time
from typing import Dict, Tuple

import websockets.sync.client
from typing_extensions import override

from openpi_client import base_policy as _base_policy
from openpi_client import msgpack_numpy


class WebsocketClientPolicy(_base_policy.BasePolicy):
    """Implements the Policy interface by communicating with a server over websocket.

    See WebsocketPolicyServer for a corresponding server implementation.
    """

    def __init__(self, host: str = "0.0.0.0", port: int = 8000) -> None:
        self._uri = f"ws://{host}:{port}"
        self._packer = msgpack_numpy.Packer()
        self._ws, self._server_metadata = self._wait_for_server()

    def get_server_metadata(self) -> Dict:
        return self._server_metadata

    def _wait_for_server(self) -> Tuple[websockets.sync.client.ClientConnection, Dict]:
        logging.info(f"Waiting for server at {self._uri}...")
        while True:
            try:
                conn = websockets.sync.client.connect(self._uri, compression=None, max_size=None)
                metadata = msgpack_numpy.unpackb(conn.recv())
                return conn, metadata
            except ConnectionRefusedError:
                logging.info("Still waiting for server...")
                time.sleep(5)

    @override
    def infer(self, obs: Dict) -> Dict:  # noqa: UP006
        data = self._packer.pack(obs)
        self._ws.send(data)
        response = self._ws.recv()
        if isinstance(response, str):
            # we're expecting bytes; if the server sends a string, it's an error.
            raise RuntimeError(f"Error in inference server:\n{response}")
        return msgpack_numpy.unpackb(response)

    def infer_with_hidden(self, obs: Dict) -> Dict:  # noqa: UP006
        """Same as ``infer`` but asks the server to also return hidden features.

        The response dict contains, in addition to the usual ``actions`` / ``state``:
          - ``cond_emb``:  (prefix_width,) mean-pooled VLM prefix features
          - ``suffix_out``: (action_horizon, expert_width) per-action-token features

        Requires the server to wrap a PyTorch pi0.5 (drifting) policy exposing
        ``infer_with_hidden``; otherwise the server returns an error.
        """
        if "_return_hidden" in obs:
            raise ValueError("`_return_hidden` is a reserved key for the client")
        payload = dict(obs)
        payload["_return_hidden"] = True
        data = self._packer.pack(payload)
        self._ws.send(data)
        response = self._ws.recv()
        if isinstance(response, str):
            raise RuntimeError(f"Error in inference server:\n{response}")
        return msgpack_numpy.unpackb(response)

    @override
    def reset(self) -> None:
        pass
