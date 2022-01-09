from workers.backend import create_backend_conf
from workers.broker import BROKER_URI

# Connection string to broker for task consumption
broker_url = BROKER_URI

# The backend store to save results in
result_backend = "db+mongodb://localhost:27017/wikiplag"
mongodb_backend_settings = create_backend_conf()

task_acks_late = True

worker_prefetch_multiplier = 1
