import logging
import os
import gdown
from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram import Router
from PIL import Image
import torch
from torchvision import models, transforms
from config import TOKEN

logging.basicConfig(level=logging.INFO)

bot = Bot(token=TOKEN)
storage = MemoryStorage()
dp = Dispatcher(storage=storage)
router = Router()


model_path = "resnet34_model.pth"
if not os.path.exists(model_path):
    url = "https://drive.google.com/uc?export=download&id=1qEj0CT2674iRFv6XTD87epwPZhstZZZQ"
    gdown.download(url, model_path, quiet=False)

class_names = ['¬πα¿µá', 'µδ»½Ñ¡«¬', '»ÑΓπσ', '¿¡ñε¬¿', 'ßΓαáπß', 'úπß∞', 'πΓ¬á']
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.resnet34(pretrained=False)
num_features = model.fc.in_features
model.fc = torch.nn.Linear(num_features, len(class_names))
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class_translation = {
    '¬πα¿µá': 'курица',
    'µδ»½Ñ¡«¬': 'цыпленок',
    '»ÑΓπσ': 'петух',
    '¿¡ñε¬¿': 'индюк',
    'ßΓαáπß': 'страус',
    'úπß∞': 'гусь',
    'πΓ¬á': 'утка'
}

@router.message(Command("start"))
async def start_handler(message: types.Message):
    await message.reply("Привет! Отправьте мне фото, и я скажу, к какому классу оно относится.")

@router.message(lambda message: message.photo)
async def photo_handler(message: types.Message):

    photo = message.photo[-1]
    file = await bot.download(photo.file_id)

    image = Image.open(file).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        _, predicted_idx = torch.max(output, 1)
        predicted_class = class_names[predicted_idx.item()]
        translated_class = class_translation.get(predicted_class, "неизвестный класс")

    await message.reply(f"Птица на данной фотографии относится к классу: *{translated_class}*", parse_mode="Markdown")

dp.include_router(router)

async def main():
    try:
        await dp.start_polling(bot)
    finally:
        await bot.session.close()

import asyncio
try:
    asyncio.run(main())
except RuntimeError:
    import nest_asyncio
    nest_asyncio.apply()
    asyncio.get_event_loop().run_until_complete(main())
