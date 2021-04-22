from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates

app = FastAPI()
templates = Jinja2Templates(directory="templates/")


@app.get('/')
def index(request: Request):
    # result = "Type a number"
    return templates.TemplateResponse('index.html', context={'request': request})


@app.get("/predict/{data}")
async def form_post(request: Request, data,):
    result = 'Kết quả: ' + data
    a= await request.form()
    print(a.keys('data'))
    return templates.TemplateResponse('index.html', context={'request': request, 'result': result})

