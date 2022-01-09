from celery import Celery

app = Celery("som")
app.config_from_object('som.config')
