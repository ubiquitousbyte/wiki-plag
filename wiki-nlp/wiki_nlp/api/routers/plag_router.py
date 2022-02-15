from typing import List

from fastapi import (
    APIRouter,
    Depends
)


from wiki_nlp.api.services.plag_service import IPlagService
from wiki_nlp.api.services.schema import Text

from wiki_nlp.api.routers import dependencies as di

router = APIRouter(prefix='/plag')


@router.post('')
def get_plags(text: Text, plag_service: IPlagService = Depends(di.get_plag_service)):
    return plag_service.find_candidates(text.text)
