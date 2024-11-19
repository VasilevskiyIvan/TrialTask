import logging
from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram import Router
from PIL import Image
import torch
from torchvision import models, transforms

# Настройки
TOKEN = "7713924720:AAFGNJQCUJeCS4kEVaQqsBxd0NpmN_v_h0E"

# Логирование
logging.basicConfig(level=logging.INFO)

# Создаем бота и диспетчер
bot = Bot(token=TOKEN)
storage = MemoryStorage()
dp = Dispatcher(storage=storage)
router = Router()

# Настройки модели
model_path = r"C:\Users\Иван\Downloads\resnet34_model.pth"  # Путь к сохраненной модели
class_names = ['¬πα¿µá', 'µδ»½Ñ¡«¬', '»ÑΓπσ', '¿¡ñε¬¿', 'ßΓαáπß', 'úπß∞', 'πΓ¬á']  # Список классов
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Загрузка модели
model = models.resnet34(pretrained=False)
num_features = model.fc.in_features
model.fc = torch.nn.Linear(num_features, len(class_names))
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)
model.eval()

# Преобразования изображения
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Словарь с русскими названиями классов
class_translation = {
    '¬πα¿µá': 'курица',
    'µδ»½Ñ¡«¬': 'цыпленок',
    '»ÑΓπσ': 'петух',
    '¿¡ñε¬¿': 'индюк',
    'ßΓαáπß': 'страус',
    'úπß∞': 'гусь',
    'πΓ¬á': 'утка'
}

# Обработчик команды /start
@router.message(Command("start"))
async def start_handler(message: types.Message):
    await message.reply("Привет! Отправьте мне фото, и я скажу, к какому классу оно относится.")

# Обработчик фотографий
@router.message(lambda message: message.photo)
async def photo_handler(message: types.Message):
    # Скачиваем фото
    photo = message.photo[-1]
    file = await bot.download(photo.file_id)

    # Открываем изображение и применяем преобразования
    image = Image.open(file).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    # Делаем предсказание
    with torch.no_grad():
        output = model(input_tensor)
        _, predicted_idx = torch.max(output, 1)
        predicted_class = class_names[predicted_idx.item()]
        translated_class = class_translation.get(predicted_class, "неизвестный класс")

    # Отправляем результат
    await message.reply(f"Птица на данной фотографии относится к классу: *{translated_class}*", parse_mode="Markdown")

# Регистрация маршрутов
dp.include_router(router)

# Основная функция запуска
async def main():
    try:
        await dp.start_polling(bot)
    finally:
        await bot.session.close()

# Запуск в Jupyter/Colab
import asyncio
try:
    asyncio.run(main())
except RuntimeError:  # Если уже есть запущенный событийный цикл
    import nest_asyncio
    nest_asyncio.apply()
    asyncio.get_event_loop().run_until_complete(main())
