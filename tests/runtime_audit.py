"""Report dependency and authentication implementations without accessing credentials."""
import importlib.metadata
import inspect
import json
import ssl
import sys
import asf_search
import hyp3_glacier_melt.doy_product
print(json.dumps({
    'python': sys.version,
    'openssl': ssl.OPENSSL_VERSION,
    'packages': {d.metadata['Name']: d.version for d in importlib.metadata.distributions()},
    'auth_source': inspect.getsource(asf_search.ASFSession.auth_with_creds),
    'exporter_path': hyp3_glacier_melt.doy_product.__file__,
}, indent=2, sort_keys=True))
