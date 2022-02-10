from fastapi import (
    APIRouter,
    Depends
)


from pydantic import BaseModel

from wiki_nlp.api.services.dmm_service import IDMMService

from wiki_nlp.api.routers import dependencies as di

router = APIRouter(prefix='/plag')


class Text(BaseModel):
    text: str


@router.get('')
def get_plags(text: Text, dmm: IDMMService = Depends(di.get_dmm_service)):
    return dmm.most_similar(text.text)
