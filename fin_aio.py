import sqlite3
from aiogram.filters import CommandStart
from aiogram.client.bot import DefaultBotProperties
from aiogram.enums import ParseMode
import aiosqlite
import base64
import json

from translate import Translator
import matplotlib.pyplot as plt  # type: ignore
import PIL  # type: ignore
import numpy as np  # type: ignore
from tensorflow import keras  # type: ignore
from tensorflow.keras import layers  # type: ignore
from tensorflow.keras.models import Sequential  # type: ignore
import pathlib
from dotenv import load_dotenv
import os
import keyboards as kb
import asyncio
import datetime
from aiogram import Bot, Dispatcher, types, F
from aiogram.types import Message, FSInputFile, file
import tensorflow as tf  # type: ignore
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.storage.memory import MemoryStorage
from gigachat import GigaChat
from aiogram.filters import Command



dataset_dir = pathlib.Path("food-101")
batch_size = 32
img_width = 180
img_height = 180
train_ds = tf.keras.utils.image_dataset_from_directory(
    dataset_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

val_ds = tf.keras.utils.image_dataset_from_directory(
    dataset_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

class_names = train_ds.class_names
print(f"Class names: {class_names}")

num_classes = len(class_names)
model = Sequential([
    tf.keras.layers.Rescaling(1. / 255, input_shape=(img_height, img_width, 3)),

    tf.keras.layers.RandomFlip("horizontal", input_shape=(img_height, img_width, 3)),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
    tf.keras.layers.RandomContrast(0.2),

    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),

    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),

    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),

    layers.Dropout(0.2),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes)
])

model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'])

model.load_weights("Foood.weights.h5")

loss, acc = model.evaluate(train_ds, verbose=2)
print("Restored model, accuracy: {:5.2f}%".format(100 * acc))



load_dotenv()
TOKEN = os.getenv('TOKEN')

def decode_credentials(encoded_str):
    decoded_bytes = base64.b64decode(encoded_str)
    decoded_str = decoded_bytes.decode('utf-8')
    client_id, client_secret = decoded_str.split(':')
    return client_id, client_secret


encoded_credentials = os.getenv('GIGA')
client_id, client_secret = decode_credentials(encoded_credentials)
GIGA = {
    'client_id': client_id,
    'client_secret': client_secret
}



credentials_str = f"{GIGA['client_id']}:{GIGA['client_secret']}"
credentials_base64 = base64.b64encode(credentials_str.encode("utf-8")).decode("utf-8")

bot = Bot(TOKEN, default=DefaultBotProperties(parse_mode=ParseMode.HTML))
storage = MemoryStorage()
dp = Dispatcher(storage=storage)


conn = sqlite3.connect('pro3.db', check_same_thread=False)
cursor = conn.cursor()


class REG(StatesGroup):
    height = State()
    age = State()
    sex = State()
    want = State()
    weight = State()
    types = State()
    length = State()
    food = State()
    food_list = State()
    food_photo = State()
    grams = State()
    food_meals = State()
    train = State()
    tren_choiser = State()
    svo = State()


async def db_table_val(user_id: int, user_age: int, user_sex: str, user_weight: float, date: str, user_aim: str,
                       imt: float, imt_str: str, cal: float, user_height: int):
    async with aiosqlite.connect('pro3.db') as conn:
        await conn.execute(
            'INSERT INTO users (user_id, user_age, user_sex, user_weight, date, user_aim, imt, imt_str, cal, user_height) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
            (user_id, user_age, user_sex, user_weight, date, user_aim, imt, imt_str, cal, user_height)
        )
        await conn.commit()


async def get_user_data(user_id: int, date: str):
    async with aiosqlite.connect('pro3.db') as conn:
        cursor = await conn.execute(
            "SELECT user_age, user_height, user_sex, user_weight, user_aim, imt, imt_str, cal FROM users WHERE user_id = ? AND date = ?",
            (user_id, date)
        )
        return await cursor.fetchone()



@dp.message(CommandStart())
async def start(message: Message):
    await message.answer_photo(
        FSInputFile(path='new_logo.jpg'),
        caption=f'Привет, {message.from_user.first_name}! Бот PROпиташка поможет тебе вести индивидуальный расчет твоего питания и активности.',
        reply_markup=kb.startMenu
    )


@dp.message(F.text == "Вход")
async def entrance(message: Message, state: FSMContext):
    user_data = await get_user_data(message.from_user.id, datetime.datetime.now().strftime('%Y-%m-%d'))
    if user_data:
        height, weight, imt, imt_using_words = user_data[1], user_data[3], user_data[5], user_data[6]
        await bot.send_message(
            message.chat.id,
            text=f'{message.from_user.first_name}, твой вес: {weight}, твой рост: {height}, твой ИМТ: {imt}, и твой вес - это {imt_using_words}.'
        )
        await message.answer(
            'Успешный вход в систему ✅. Теперь тебе доступен весь функционал бота, достаточно просто выбрать желаемую функцию в клавиатуре)',
            reply_markup=kb.main_menu)


    else:
        await bot.send_message(message.chat.id, text='Твоих данных в базе нет 🙁 Для начала пройди регистрацию', reply_markup=kb.reRig)


@dp.message(F.text == 'Регистрация')
async def registration(message: Message, state: FSMContext):
    await db_table_val(
        user_id=message.from_user.id,
        date=datetime.datetime.now().strftime('%Y-%m-%d'),
        user_aim="",
        imt=0.0,
        imt_str="",
        cal=0.0,
        user_sex="",
        user_height=0,
        user_weight=0.0,
        user_age=0
    )
    await state.set_state(REG.height)
    await bot.send_message(message.chat.id, text='Введи свой рост в сантиметрах:')


@dp.message(REG.height)
async def height(message: Message, state: FSMContext):
    await state.update_data(height=float(message.text))
    await state.set_state(REG.age)
    await message.answer('Введи свой возраст:')

@dp.message(REG.age)
async def age(message: Message, state: FSMContext):
    await state.update_data(age=int(message.text))
    await state.set_state(REG.sex)
    await message.answer('Выбери свой пол:', reply_markup=kb.sex)

@dp.message(REG.sex)
async def sex(message: Message, state: FSMContext):
    await state.update_data(sex=message.text)
    await state.set_state(REG.want)
    await message.answer('Выбери, как хочешь измениться', reply_markup=kb.want)

@dp.message(REG.want)
async def want(message: Message, state: FSMContext):
    await state.update_data(want=message.text)
    await state.set_state(REG.weight)
    await message.answer('Введи свой вес в килограммах', reply_markup=types.ReplyKeyboardRemove())

@dp.message(REG.weight)
async def wei(message: Message, state: FSMContext):
    await state.update_data(weight=message.text)
    data = await state.get_data()
    height, sex, age, weight, aim = data['height'], data['sex'], data['age'], data['weight'], data['want']


    if "," in weight:
        we1 = message.text.split(",")
        weight = int(we1[0]) + int(we1[1]) / 10 ** len(we1[1])
    else:
        weight = float(message.text)


    height, sex, age = int(height), str(sex), int(age)
    imt = round(weight / ((height / 100) ** 2), 3)
    imt_using_words = calculate_imt_description(imt)
    cal = calculate_calories(sex, weight, height, age)


    cursor.execute(
        f"UPDATE users SET user_weight = ?, imt = ?, imt_str = ?, cal = ?, user_sex = ?, user_age = ?, user_height = ?, user_aim = ? WHERE user_id = ? AND date = ?",
        (weight, imt, imt_using_words, cal, sex, age, height, aim, message.from_user.id, datetime.datetime.now().strftime('%Y-%m-%d'))
    )
    conn.commit()
    await bot.send_message(
        message.chat.id,
        text=f'{message.from_user.first_name}, твой вес: {weight}, твой рост: {height}, твой индекс массы тела: {imt}, и твой вес - это {imt_using_words}.'
        )
    await message.answer('Регистрация пройдена успешно ✅. Теперь тебе доступен весь функционал бота, достаточно просто выбрать желаемую функцию в клавиатуре)', reply_markup=kb.main_menu)
    await state.clear()


async def calculate_imt_description(imt):
    if round(imt) < 15:
        return 'сильно меньше нормы'
    elif round(imt) in range(14, 18):
        return 'Недостаточная масса'
    elif round(imt) in range(18, 25):
        return 'Норма'
    elif round(imt) in range(25, 30):
        return 'Предожирение'
    else:
        return 'Ожирение'

async def calculate_calories(sex, weight, height, age):
    if sex == 'Мужчина':
        return (10 * weight) + (6.25 * height) - (5 * age) + 5
    elif sex == 'Женщина':
        return (10 * weight) + (6.25 * height) - (5 * age) - 161
    return 0

def split_message(text, max_length=4096):
    """Разделяем текст на части не более max_length символов."""
    return [text[i:i + max_length] for i in range(0, len(text), max_length)]

def is_not_none(item):
    return item is not None





async def generate(message, zap):
    try:
        async with GigaChat(
            credentials=os.getenv('GIGA'),
            verify_ssl_certs=False) as giga:
              pit= giga.chat(zap)
              return pit.choices[0].message.content
    except Exception as e:
        print(f"Ошибка при генерации плана: {str(e)}")
        return f"Ошибка при генерации плана: {e}"

@dp.message(F.text == 'Добавить тренировки')
async def tren(message: Message, state: FSMContext):
    await bot.send_message(message.chat.id, text='Какая была тренировка:', reply_markup=kb.tren)
    await state.set_state(REG.types)
@dp.message(REG.types)
async def tren_type(message: Message, state: FSMContext):
    await state.update_data(types=message.text)
    await state.set_state(REG.length)
    await message.answer(text = 'Хорошо, а введи, сколько минут длилась твоя тренировка:')
@dp.message(REG.length)
async def tren_len(message: Message, state: FSMContext):
    await state.update_data(length=message.text)
    data = await state.get_data()
    cursor.execute("SELECT user_weight FROM users WHERE date = ? AND user_id = ?",
                   (datetime.datetime.now().strftime('%Y-%m-%d'), message.from_user.id))
    weight = float(cursor.fetchone()[-1])
    time = int(data['length'])
    intensivity = float()
    if data['types'] == 'Лёгкая':
        intensivity = 2.5
    if data['types'] == 'Умеренная':
        intensivity = 3
    if data['types'] == 'Тяжёлая':
        intensivity = 3.5
    tren_cal = round((weight * intensivity * time / 24), 3)
    await bot.send_message(message.chat.id, text=f'Прекрасно! Ты за тренировку сжёг {tren_cal} килокалорий. Так держать!!',
                     reply_markup=kb.main_menu)
    async with aiosqlite.connect('pro3.db') as conn:
        await conn.execute(
    'INSERT INTO user_training_cal (user_id, date, user_train_cal, tren_time) VALUES (?, ?, ?, ?)',
                   (message.from_user.id, datetime.datetime.now().strftime('%Y-%m-%d'), tren_cal, time))
        await conn.commit()

    cursor.execute("SELECT SUM(user_train_cal) FROM user_training_cal WHERE date = ? AND user_id = ?",
            (datetime.datetime.now().strftime('%Y-%m-%d'), message.from_user.id))
    result = cursor.fetchone()[0]


    await bot.send_message(message.chat.id,
                     text=f'{message.from_user.first_name}, за сегодня ты сжёг {result} килокалорий. Так держать!!')
    await state.clear()





@dp.message(F.text == 'Ввести еду за день')
async def food1(message: Message, state: FSMContext):
    await message.answer(text='Выбери, как ты хотел бы добавить сьеденную пищу?', reply_markup=kb.food)
    await state.set_state(REG.food)


@dp.message(REG.food)
async def foodchoise(message: Message, state: FSMContext):
    await state.update_data(food=message.text)
    await state.set_state(REG.food_photo)

    data = await state.get_data()
    if data['food'] == 'С помощью текста':
        await message.answer(text='Введи через запятую названия блюд:', reply_markup=types.ReplyKeyboardRemove())
        await state.set_state(REG.food_list)

    if data['food'] == 'С помощью фото':
        await message.answer(text='Пришли фотографию для распознавания', reply_markup=types.ReplyKeyboardRemove())
        await state.set_state(REG.food_photo)

@dp.message(REG.food_list)
async def names(message:Message, state: FSMContext):
    await state.update_data(food_list = message.text.replace(" ", "").split(","))
    await bot.send_message(message.chat.id, text="Введи, сколько грамм было в этом блюде")
    await state.set_state(REG.grams)



@dp.message(REG.food_photo)
async def handle_photo(message:Message, state: FSMContext):
    await state.update_data(food_photo=message.photo)
    data = await state.get_data()
    photo = data['food_photo'][-1]

    await state.clear()
    name_a = []
    file_info = await bot.get_file(photo.file_id)
    downloaded_file = await bot.download_file(file_info.file_path)
 #   downloaded_file = await bot.download_file(file_path, file_info)
    save_path = 'photo.jpg'
    with open(save_path, 'wb') as new_file:
         new_file.write(downloaded_file.read())
    await bot.send_message(message.chat.id, 'Фото схранено')
    img =  tf.keras.utils.load_img("photo.jpg", target_size=(img_height, img_width))
    img_array =  tf.keras.utils.img_to_array(img)
    img_array =  tf.expand_dims(img_array, 0)
    predictions = model.predict(img_array)
    score =  tf.nn.softmax(predictions[0])
    lol =  str(class_names[np.argmax(score)])
    translator =  Translator(from_lang="en", to_lang="ru")
    name_a.append(translator.translate(lol).title())
    await state.set_state(REG.grams)
    await bot.send_message(message.chat.id, text="Введи, сколько грамм было в этом блюде")
    await state.update_data(food_list=name_a)

@dp.message(F.text == 'Присоедениться к чату')
async def chat(message:Message):
    await message.answer(text = 'https://t.me/+QVhMA2topDgzOWVi', reply_markup=kb.main_menu)


@dp.message(F.text == 'Добавить выпитый стаканчик воды')
async def chat(message:Message):
    async with aiosqlite.connect('pro3.db') as conn:
        await conn.execute(
    'INSERT INTO water (user_id, date, count) VALUES (?, ?, ?)',
                   (message.from_user.id, datetime.datetime.now().strftime('%Y-%m-%d'), 1))
        await conn.commit()
    await message.answer(text = 'Стакан добавлен', reply_markup=kb.main_menu)

@dp.message(REG.grams)
async def grams(message:Message, state: FSMContext):
    await state.update_data(grams=message.text)
    data = await state.get_data()
    gram = data['grams'].split(",")
    name_a =  data['food_list']
    print(name_a, data, gram)
    for m in range(len(name_a)):
        with open('products.json') as f:
            file_content = f.read()
            foods = json.loads(file_content)
            for i in range(len(foods)):
                if foods[i]["name"] == name_a[m].title():
                    b = round(float(foods[i]["bgu"].split(',')[0]) * float(gram[m]) / 100, 3)
                    g = round(float(foods[i]["bgu"].split(',')[1]) * float(gram[m]) / 100, 3)
                    u = round(float(foods[i]["bgu"].split(',')[2]) * float(gram[m]) / 100, 3)
                    food_cal =  float(foods[i]["kcal"]) * float(gram[m]) / 100
                    print(b, g, u, food_cal)
                    cursor.execute(
                        'INSERT INTO user_pit (user_id, date, user_name_of_food, b, g, u, food_cal) VALUES (?, ?, ?, ?, ?, ?, ?)',
                        (message.from_user.id,datetime.datetime.now().strftime('%Y-%m-%d') , name_a[m], b, g, u, food_cal))
                    conn.commit()
    await bot.send_message(message.chat.id, text='Данные записаны)', reply_markup=kb.main_menu)
    await state.clear()


@dp.message(F.text == 'Недельный план пиания и тренировок')
async def ai(message: Message, state: FSMContext):
    await message.answer( text='Подожди пару секунд, нейросеть генерирует ответ)')

    cursor.execute(
        "SELECT user_aim, cal, user_sex, user_age, imt, user_weight, user_height FROM users WHERE date = ? AND user_id = ?",
        (datetime.datetime.now().strftime('%Y-%m-%d'), message.from_user.id)
    )
    aim, cal, sex, age, imt, weight, height = cursor.fetchone()
    zap_pit=  f"Придумай индивидуальный план разнообразного питания на неделю для {sex},чей рост равен {height}, возраст равен {age}, имт равен {imt} и цель {aim}"
    plan_pit= await generate(message, zap_pit)
    zap_tren= f"Придумай индивидуальный план тренировок на неделю для {sex}, чей рост равен {height}, возраст равен {age},  чей имт равен {imt} , чья цель {aim} и чей индивидуальный план питания {plan_pit}"
    plan_train = await generate(message, zap_tren)
    try:
        if plan_pit and plan_train:
            # Разделяем длинные сообщения на части
            for part in split_message(plan_pit):
                await bot.send_message(message.chat.id, text=part)
            for part in split_message(plan_train):
                await bot.send_message(message.chat.id, text=part)
            await message.answer(
                text='Выданные планы питания и тренировок являются лишь рекоменданиями, которые ты можешь выполнять по желанию. ',
                reply_markup=kb.main_menu)

        else:
            await bot.send_message(message.chat.id, text="Не удалось получить данные пользователя.", reply_markup=kb.main_menu)
    except Exception as e:
        print(f"Ошибка при генерации плана: {str(e)}")
        return f"Ошибка при генерации плана: {e}"


@dp.message(F.text == 'Помочь с рецептом')
async def ai_food(message: Message, state: FSMContext):
    await message.answer(text = 'Выбери, для какого приёма пищи необходимо сгенерировать рецепт:', reply_markup=kb.meals)
    await state.set_state(REG.food_meals)

@dp.message(REG.food_meals)
async def ai_food_meals(message: Message, state: FSMContext):
    await state.update_data(food_meals=message.text)
    data = await state.get_data()
    meal = data['food_meals']
    zap = f"Придумай новый рецепт {meal} с рекомендациями по готовке"
    await message.answer( text='Подожди пару секунд, нейросеть генерирует ответ)')
    plan_pit= await generate(message, zap)
    try:
        if plan_pit:
            # Разделяем длинные сообщения на части
            for part in split_message(plan_pit):
                await bot.send_message(message.chat.id, text=part, reply_markup=kb.main_menu)
        else:
            await bot.send_message(message.chat.id, text="Не удалось получить данные пользователя.",
                                   reply_markup=kb.main_menu)
    except Exception as e:
        print(f"Ошибка при генерации плана: {str(e)}")
        return f"Ошибка при генерации плана: {e}"
    await state.clear()


@dp.message(F.text == 'Помочь с тренировкой')
async def ai_food(message: Message, state: FSMContext):
    await message.answer(text = 'Выбери, какую тренировку ты хотел бы провести?', reply_markup=kb.tren_type)
    await state.set_state(REG.train)

@dp.message(REG.train)
async def train(message: Message, state: FSMContext):
    await state.update_data(train=message.text)
    data = await state.get_data()
    type_tren = data['train']
    await state.clear()
    cursor.execute("SELECT imt FROM users WHERE date = ? AND user_id = ?",
                   (datetime.datetime.now().strftime('%Y-%m-%d'), message.from_user.id))
    imt = float(cursor.fetchone()[-1])
    zap = f"Придумай {type_tren} тренировку с рекомендациями для человека с ИМТ равным {imt}"
    await message.answer( text='Подожди пару секунд, нейросеть генерирует ответ)')
    tren = await generate(message, zap)
    try:
        if tren:
            # Разделяем длинные сообщения на части
            for part in split_message(tren):
                await bot.send_message(message.chat.id, text=part, reply_markup=kb.tren_choise )
                await state.set_state(REG.tren_choiser)

        else:
            await bot.send_message(message.chat.id, text="Не удалось получить данные пользователя.",
                                   reply_markup=kb.main_menu)
    except Exception as e:
        print(f"Ошибка при генерации плана: {str(e)}")
        return f"Ошибка при генерации плана: {e}"

@dp.message(REG.tren_choiser)
async def choising(message: Message, state: FSMContext):
    await state.update_data(tren_choiser=message.text)
    data = await state.get_data()
    mes = data['tren_choiser']
    await state.clear()
    if mes == 'Вернуться в меню':
        await message.answer(text = 'Хорошо',reply_markup=kb.main_menu)
    else:
        await message.answer(text = 'Функция пока не доступна((', reply_markup=kb.main_menu)

@dp.message(F.text == 'Вход в програму')
async def ais(message: Message, state: FSMContext):
    await message.answer( text='Теперь ты можешь вводить продукты, которые ты сегодня употребил и тренировки, которые ты сегодня прошёл, а в конце дня ты будешь получать отчёт по твоим Б/Ж/У за день и затраченным калориям',reply_markup=kb.main_menu
)

@dp.message(F.text == 'Сводка')
async def svod(message: Message, state: FSMContext):
    await message.answer(text = 'Выбери, за какой период ты хочешь увидеть сводку:', reply_markup=kb.svo)
    await state.set_state(REG.svo)
@dp.message(REG.svo)
async def svodka(message: Message, state: FSMContext):
    await state.update_data(tren_choiser=message.text)
    data = await state.get_data()
    mes = data['tren_choiser']
    await state.clear()
    if mes == 'День':
        cursor.execute("SELECT SUM(user_train_cal) FROM user_training_cal WHERE date = ? AND user_id = ?",
                       (datetime.datetime.now().strftime('%Y-%m-%d'), message.from_user.id))
        result_tren = cursor.fetchone()
        col_call_tren = result_tren[0]
        cursor.execute("SELECT SUM(food_cal) FROM user_pit WHERE date = ? AND user_id = ?",
                       (datetime.datetime.now().strftime('%Y-%m-%d'), message.from_user.id))
        result_cal_food = cursor.fetchone()
        col_cal_food = result_cal_food[0]
        cursor.execute("SELECT SUM(b) FROM user_pit WHERE date = ? AND user_id = ?",
                       (datetime.datetime.now().strftime('%Y-%m-%d'), message.from_user.id))
        result_b = cursor.fetchone()
        col_b = round(result_b[0], 3)
        cursor.execute("SELECT SUM(g) FROM user_pit WHERE date = ? AND user_id = ?",
                       (datetime.datetime.now().strftime('%Y-%m-%d'), message.from_user.id))
        result_g = cursor.fetchone()
        col_g = round(result_g[0], 3)
        cursor.execute("SELECT SUM(u) FROM user_pit WHERE date = ? AND user_id = ?",
                       (datetime.datetime.now().strftime('%Y-%m-%d'), message.from_user.id))
        result_u = cursor.fetchone()
        col_u = round(result_u[0], 3)
        cursor.execute("SELECT SUM(count) FROM water WHERE date = ? AND user_id = ?",
                       (datetime.datetime.now().strftime('%Y-%m-%d'), message.from_user.id))
        result_wat = cursor.fetchone()
        col_wat = round(result_wat[0], 3)
        cursor.execute("SELECT user_name_of_food FROM user_pit WHERE date = ? AND user_id = ?",
                       (datetime.datetime.now().strftime('%Y-%m-%d'), message.from_user.id))
        ff = ''
        result_ff = cursor.fetchall()
        for i in result_ff:
            ff += str(i[0])
            ff += ', '

        await bot.send_message(message.chat.id, text=f"""
{message.from_user.first_name}, за {datetime.datetime.now().strftime('%Y-%m-%d')} у тебя такие результаты:
во время тренировок ты сбросил {round(col_call_tren, 3) if col_call_tren else 0} килокалорий, 
сегодняшний твой рацион состоял из {ff} калорийность рациона составила {round(col_cal_food, 3) if col_cal_food else 0}, 
за день было употреблено Б/Ж/У в соотношении {col_b if col_b else 0}/{col_g if col_g else 0}/{col_u if col_u else 0} и выпито {col_wat * 300 if col_wat else 0} милилитров воды. Я верю в твой успех!!""",
                         reply_markup=kb.main_menu)
    elif mes == 'Месяц':
        weight_month = []
        sr_b = []
        sr_g = []
        sr_u = []
        sr_cal = []
        sr_w = []
        sr_tren = []
        for i in range(1, 32):
            datee = f'{str(datetime.datetime.now().year)}-{str(datetime.datetime.now().month).zfill(2)}-{str(i).zfill(2)}'
            cursor.execute("SELECT user_weight FROM users WHERE user_id = ? AND date = ?",
                           (message.from_user.id, datee))
            weight_data = cursor.fetchall()
            if weight_data:
                weight_month.append(weight_data)
            cursor.execute("SELECT sum(b) FROM user_pit WHERE user_id = ? AND date = ?",
                           (message.from_user.id, datee))
            b_data = cursor.fetchone()
            if b_data:
                sr_b.append(b_data[0])
            cursor.execute("SELECT sum(g) FROM user_pit WHERE user_id = ? AND date = ?",
                           (message.from_user.id, datee))
            g_data = cursor.fetchone()
            if g_data:
                sr_g.append(g_data[0])
            cursor.execute("SELECT sum(u) FROM user_pit WHERE user_id = ? AND date = ?",
                           (message.from_user.id, datee))
            u_data = cursor.fetchone()
            if u_data:
                sr_u.append(u_data[0])
            cursor.execute("SELECT sum(count) FROM water WHERE user_id = ? AND date = ?",
                           (message.from_user.id, datee))
            w_data = cursor.fetchone()
            if w_data:
                sr_w.append(w_data[0])
            cursor.execute("SELECT sum(user_train_cal) FROM user_training_cal WHERE user_id = ? AND date = ?",
                           (message.from_user.id, datee))
            cal_data = cursor.fetchone()
            if cal_data:
                sr_cal.append(cal_data[0])
            cursor.execute("SELECT sum(tren_time) FROM user_training_cal WHERE user_id = ? AND date = ?",
                           (message.from_user.id, datee))
            time_data = cursor.fetchone()
            if time_data:
                sr_tren.append(time_data[0])
        if weight_month and sr_b and sr_g and sr_u and sr_cal and sr_tren and sr_w:
            weig_1 = weight_month[0][0]
            weig_2 = weight_month[-1][-1]
            new_sr_b = list(filter(is_not_none, sr_b))
            new_sr_g = list(filter(is_not_none, sr_g))
            new_sr_u = list(filter(is_not_none, sr_u))
            new_sr_w = list(filter(is_not_none, sr_w))
            new_sr_cal = list(filter(is_not_none, sr_cal))
            new_sr_tren = list(filter(is_not_none, sr_tren))
            if sum(new_sr_b) > 0:
                avg_b = round(sum(new_sr_b) / len(new_sr_b), 3)
            else:
                avg_b = 0
            if sum(new_sr_g) > 0:
                avg_g = round(sum(new_sr_g) / len(new_sr_g), 3)
            else:
                avg_g = 0
            if sum(new_sr_u) > 0:
                avg_u = round(sum(new_sr_u) / len(new_sr_u), 3)
            else:
                avg_u = 0
            if sum(new_sr_w) > 0:
                avg_w = sum(new_sr_w) / len(new_sr_w) * 300
            else:
                avg_w = 0

            avg_training_time = round(sum(new_sr_tren) / len(new_sr_tren), 3) if round(
                sum(new_sr_tren) / len(new_sr_tren), 3) else 0  # Расчет среднего времени тренировок
            avg_calories_burned = round(sum(new_sr_cal) / len(new_sr_cal), 3) if round(
                sum(new_sr_cal) / len(new_sr_cal), 3) else 0  # Расчет среднего числа сожжённых калорий
            await bot.send_message(message.chat.id, text=f"""{message.from_user.first_name}, за месяц произошли такие изменения:
твой вес изменился с {weig_1[0]} на {weig_2[0]}, 
в день ты тренировался {avg_training_time} минут каждый день, сжигая при этом в среднем {avg_calories_burned} килокалорий,
в день твои Б/Ж/У были в соотношении {avg_b}/{avg_g}/{avg_u} и выпивалось около {avg_w} милилитров воды.Я в тебя верю!""",
                             reply_markup=kb.main_menu)
        else:
            await bot.send_message(message.chat.id, "Нет данных за этот месяц.")
    elif mes == 'Год':
        all_data = []
        total_food_cal = 0
        total_b = 0
        total_g = 0
        total_u = 0
        total_w = 0
        weight_data_all = []
        food_months_with_data = set()

        current_date = datetime.datetime.now()

        for i in range(12):
            current_month = current_date.month - i
            current_year = current_date.year

            if current_month <= 0:
                current_year -= 1
                current_month += 12

            first_day_of_month = datetime.date(current_year, current_month, 1)
            if current_month == 12:
                last_day_of_month = datetime.date(current_year + 1, 1, 1) - datetime.timedelta(days=1)
            else:
                last_day_of_month = datetime.date(current_year, current_month + 1, 1) - datetime.timedelta(days=1)

            cursor.execute("""
                    SELECT SUM(food_cal), SUM(b), SUM(g), SUM(u)
                    FROM user_pit 
                    WHERE date >= ? AND date <= ? AND user_id = ?
                    GROUP BY strftime('%Y-%m', date)
                """, (
                first_day_of_month.strftime('%Y-%m-%d'), last_day_of_month.strftime('%Y-%m-%d'), message.from_user.id))
            result_food = cursor.fetchone()
            cursor.execute("""
                          SELECT SUM(count)
                          FROM water
                          WHERE date >= ? AND date <= ? AND user_id = ?
                          GROUP BY strftime('%Y-%m', date)
                      """, (
                first_day_of_month.strftime('%Y-%m-%d'), last_day_of_month.strftime('%Y-%m-%d'), message.from_user.id))
            result_wat = cursor.fetchone()
            if result_food and result_food[0]:
                all_data.append(result_food)
                total_food_cal += result_food[0]
                total_b += result_food[1]
                total_g += result_food[2]
                total_u += result_food[3]
                food_months_with_data.add((current_year, current_month))
            if result_wat and result_wat[0]:
                total_w += result_wat[0]
            cursor.execute("""
                    SELECT date, user_weight 
                    FROM users 
                    WHERE date >= ? AND date <= ? AND user_id = ?
                    ORDER BY date ASC
                """, (
                first_day_of_month.strftime('%Y-%m-%d'), last_day_of_month.strftime('%Y-%m-%d'), message.from_user.id))
            weight_data = cursor.fetchall()

            if weight_data:
                weight_data_all.extend(weight_data)

        if weight_data_all:
            weight_data_all.sort(key=lambda x: x[0])
            start_weight = weight_data_all[0][1]
            end_weight = weight_data_all[-1][1]
        else:
            start_weight = 'нет данных'
            end_weight = 'нет данных'

        cursor.execute("""
                SELECT AVG(user_train_cal) 
                FROM user_training_cal 
                WHERE user_id = ?
            """, (message.from_user.id,))
        result_train = cursor.fetchone()
        avg_train_cal = result_train[0] if result_train and result_train[0] else 0

        avg_food_cal = total_food_cal / len(food_months_with_data) if food_months_with_data else 0
        avg_b = round(total_b / len(food_months_with_data), 3) if food_months_with_data else 0
        avg_g = round(total_g / len(food_months_with_data), 3) if food_months_with_data else 0
        avg_u = round(total_u / len(food_months_with_data), 3) if food_months_with_data else 0
        all_data = list(filter(is_not_none, all_data))
        await bot.send_message(message.chat.id, text=f"""
За последние 12 месяцев:
твой вес изменился с {start_weight} на {end_weight};
в среднем за месяц ты сжигал {avg_train_cal:.1f} килокалорий на тренировках;
в  среднем твой рацион составлял {avg_food_cal:.2f} килокалорий
в начале года твой рацион составлял {round(float(all_data[-1][0]), 3) if round(float(all_data[-1][0]), 3) else 0} килокаларий , а конечное значение {round(float(all_data[0][0]), 3) if round(float(all_data[0][0]), 3) else 0} килокалорий,
соотношение Б/Ж/У в среднем: {avg_b:.2f}/{avg_g:.2f}/{avg_u:.2f}, 
в начале года твои Б/Ж/У составляли соотношение {round(all_data[-1][1], 3) if round(all_data[-1][1], 3) else 0}/{round(all_data[-1][2], 3) if round(all_data[-1][2], 3) else 0}/{round(all_data[-1][3], 3) if round(all_data[-1][3], 3) else 0} , а конечное значение {round(all_data[0][1], 3) if round(all_data[0][1], 3) else 0}/{round(all_data[0][2], 3) if round(all_data[0][2], 3) else 0}/{round(all_data[0][3], 3) if round(all_data[0][3], 3) else 0},
и было выпито в среднем {total_w / len(food_months_with_data) * 300 if total_w / len(food_months_with_data) * 300 else 0} милилитров воды каждый день.""",
                         reply_markup=kb.main_menu)


async def main():

    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
