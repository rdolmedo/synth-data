"""Agent-only offline protocol. Evaluator model clients, setup, rendering and grading remain on the host."""
OFFLINE_NOTICE = ('\n\nEnvironment policy: internet access is disabled. Use the supplied reference files and local tools; '
                  'web fetch/search tools are unavailable.')
POLICY = {'mode': 'offline', 'enforcement': 'owned-user-and-network-namespace-no-uplink',
          'web_tools': False, 'version': 1}


def require_offline(cfg):
    if cfg.get('agent_network') != POLICY:
        raise ValueError('This code runs only explicitly recorded offline agent runs. Legacy/mismatched network '
                         'protocols require a new run ID; stored results are not relabeled.')
