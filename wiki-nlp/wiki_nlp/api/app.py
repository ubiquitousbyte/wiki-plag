from fastapi import (
    FastAPI
)
#import uvicorn
#from torch.multiprocessing import set_start_method

from wiki_nlp.api.routers import plag_router


app = FastAPI(title='Plagiarism Detector', version='0.1.0')

app.include_router(plag_router.router, prefix='/api/v1')

"""if __name__ == '__main__':
    try:
        set_start_method("spawn")
    except:
        pass

    uvicorn.run(app, host='0.0.0.0', port=8082, log_level='debug')"""
