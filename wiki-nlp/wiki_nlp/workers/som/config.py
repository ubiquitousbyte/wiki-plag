from workers.backend import create_backend_conf
from workers.broker import BROKER_URI

broker_url = BROKER_URI
result_backend = "db+mongodb://localhost:27017/wikiplag"
mongodb_backend_settings = create_backend_conf()
