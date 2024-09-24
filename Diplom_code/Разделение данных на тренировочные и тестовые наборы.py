#Импорт нужных библиотек
import glob, os
from math import *
from tqdm import tqdm
import shutil


#Импорт путей к каталогам где содержатся данные
input_folders = [
    'dataset/Cat/',
    'dataset/Dog/'
]

#Обьявление папок куда будут сохранятся результаты
BASE_DIR_ABSOLUTE = 'D:\\Python\\PycharmProjects\\neiro\\CatDogNeiro'
OUT_DIR = './dataset/CatDog_prepared/'

OUT_TRAIN = OUT_DIR + 'train/'
OUT_VAL = OUT_DIR + 'test/'

#Коэффицент разделения данных
coeff = [80,20]
exceptions = ['classes']

#Проверка на корректность введенных коэффицентов
if int(coeff[0]) + int(coeff[1]) > 100:
    print("Coeff can't exceed 100%.")
    exit(1)

#Создание списка
def chunker(seq,size):
    return (seq[pos:pos + size] for pos in range(0,len(seq),size))

print(f"Preparing  images data by {coeff[0]/coeff[1]} rule.")
print(f"Sourse folders: {len(input_folders)} ")
print("Gathering data ...")

#Основное тело работы программы
sourse = {}
for sf in input_folders:
    sourse.setdefault(sf,[])

    os.chdir(BASE_DIR_ABSOLUTE)
    os.chdir(sf)

    for filename in glob.glob("*.jpg"):
        sourse[sf].append(filename)

train = {}
val={}
for sk, sv in sourse.items():
    chunks = 10
    train_chunk = floor(chunks * (coeff[0]/ 100))
    val_chunk = chunks - train_chunk

    train.setdefault(sk, [])
    val.setdefault(sk, [])
    for item in chunker(sv, chunks):
        train[sk].extend(item[0:train_chunk])
        val[sk].extend(item[train_chunk:])

train_sum = 0
val_sum = 0

for sk,sv in train.items():
    train_sum+=len(sv)

for sk,sv in val.items():
    val_sum+=len(sv)

print(f'\nOverall TRAIN images count: {train_sum}')
print(f'Overall TEST images count: {val_sum}')

os.chdir(BASE_DIR_ABSOLUTE)
print("\nCoping TRAIN sourse items ot prepered folder ...")
for sk,sv in tqdm(train.items()):
    for item in tqdm(sv):
        imgfile_sourse = sk + item
        imgdile_dest = OUT_TRAIN+ sk.split('/')[-2] + '/'

        os.makedirs(imgdile_dest,exist_ok=True)
        shutil.copyfile(imgfile_sourse, imgdile_dest + item)

os.chdir(BASE_DIR_ABSOLUTE)
print('\nCopying VAL sourse items to prepared folder ...')
for sk,sv in tqdm(val.items()):
    for item in tqdm(sv):
        imgfile_sourse = sk + item
        imgdile_dest = OUT_VAL+ sk.split('/')[-2] + '/'

        os.makedirs(imgdile_dest,exist_ok=True)
        shutil.copyfile(imgfile_sourse, imgdile_dest + item)

print('\nDONE')







#для деления датасета из чисел в формате сvs









# Импорт нужных библиотек
import os
import pandas as pd
from math import floor
from tqdm import tqdm
import shutil

# Импорт путей к файлам данных
input_files = [
    'dataset/data1.npz',
    'dataset/data2.npz'
]

# Объявление папок, куда будут сохраняться результаты
OUT_DIR = './dataset/data_prepared/'

OUT_TRAIN = OUT_DIR + 'train/'
OUT_VAL = OUT_DIR + 'test/'

# Коэффициент разделения данных
coeff = [80, 20]

# Проверка на корректность введенных коэффициентов
if int(coeff[0]) + int(coeff[1]) > 100:
    print("Coeff can't exceed 100%.")
    exit(1)

# Создание списка
def chunker(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))

print(f"Preparing data by {coeff[0]}/{coeff[1]} rule.")
print(f"Source files: {len(input_files)} ")
print("Gathering data ...")

# Основное тело работы программы
source = {}
for file in input_files:
    source[file] = pd.read_csv(file)

train = {}
val = {}
for file, data in source.items():
    data_len = len(data)
    train_size = floor(data_len * (coeff[0] / 100))
    val_size = data_len - train_size

    train[file] = data.iloc[:train_size]
    val[file] = data.iloc[train_size:]

train_sum = sum(len(df) for df in train.values())
val_sum = sum(len(df) for df in val.values())

print(f'\nOverall TRAIN data count: {train_sum}')
print(f'Overall TEST data count: {val_sum}')

# Создание папок для хранения результатов
os.makedirs(OUT_TRAIN, exist_ok=True)
os.makedirs(OUT_VAL, exist_ok=True)

print("\nSaving TRAIN data to prepared folder ...")
for file, data in tqdm(train.items()):
    base_filename = os.path.basename(file)
    dest_file = os.path.join(OUT_TRAIN, base_filename)
    data.to_csv(dest_file, index=False)

print('\nSaving TEST data to prepared folder ...')
for file, data in tqdm(val.items()):
    base_filename = os.path.basename(file)
    dest_file = os.path.join(OUT_VAL, base_filename)
    data.to_csv(dest_file, index=False)

print('\nDONE')